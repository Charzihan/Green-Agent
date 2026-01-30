"""
Functional tests for refactored green agent.

These tests verify that:
1. Green agent correctly delegates to purple agent
2. Purple agent returns single label (handles voting internally)
3. Accuracy calculation works correctly
4. Only 1 call to purple per row (not 3)
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Add parent directory to path for test_agent imports
sys.path.append(str(Path(__file__).resolve().parent))

from test_agent import send_text_message


def _get_free_port() -> int:
    """Find an available port for mock purple server"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout_s: float = 10.0) -> None:
    """Wait for server to be ready by polling agent card endpoint"""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = httpx.get(f"{url}/.well-known/agent-card.json", timeout=1.5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server not ready: {url}")


@pytest.fixture()
def mock_purple():
    """
    Start mock purple server on a free port.
    
    This mock purple simulates the refactored behavior:
    - Receives 1 request per evaluation
    - Returns single label (internally "votes" if needed)
    """
    port = _get_free_port()
    url = f"http://127.0.0.1:{port}"

    # Use the updated mock purple server (v2)
    script = Path(__file__).parent / "mock_purple_server.py"
    proc = subprocess.Popen(
        [sys.executable, str(script), "--host", "127.0.0.1", "--port", str(port), "--card-url", f"{url}/"],
    )

    try:
        _wait_ready(url)
        # Reset counter before tests
        httpx.post(f"{url}/debug/reset", timeout=2)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _extract_result_data(events):
    """
    Extract DataPart from task artifacts.
    
    Handles both streaming and non-streaming responses.
    """
    for event in events:
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event
            
            # Check task.artifacts directly (non-streaming)
            if hasattr(task, 'artifacts') and task.artifacts:
                for artifact in task.artifacts:
                    for part in getattr(artifact, 'parts', []) or []:
                        root = getattr(part, 'root', None)
                        if root and hasattr(root, 'data'):
                            return root.data
            
            # Also check update.artifact (streaming)
            if update:
                kind = getattr(update, 'kind', None)
                if kind == "artifact-update":
                    artifact = getattr(update, 'artifact', None)
                    if artifact:
                        for part in getattr(artifact, 'parts', []) or []:
                            root = getattr(part, 'root', None)
                            if root and hasattr(root, 'data'):
                                return root.data
    
    return None


@pytest.mark.asyncio
async def test_green_single_call_per_row(agent, mock_purple, tmp_path):
    """
    Test that green agent makes exactly 1 call to purple per row.
    
    This is the key difference from the old implementation (which made 3 calls).
    """
    csv_path = tmp_path / "single_row.csv"
    csv_path.write_text(
        "ruleset,trad_result\n"
        "\"TESTCASE=MAJORITY\\nrule \\\"A\\\" when Item X changed then end\",SAC\n",
        encoding="utf-8",
    )

    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "dataset_path": str(csv_path),
            "ruleset_column": "ruleset",
            "gold_column": "trad_result",
            "max_rows": 1,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None, "Green agent did not emit a data artifact"

    # Verify accuracy
    assert data["metrics"]["rows_evaluated"] == 1
    assert data["metrics"]["accuracy"] == 1.0

    # KEY ASSERTION: Only 1 call to purple (not 3)
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 1, f"Expected 1 purple call for 1 row, got {calls}"


@pytest.mark.asyncio
async def test_green_multiple_rows(agent, mock_purple, tmp_path):
    """
    Test that green agent makes 1 call per row for multiple rows.
    """
    csv_path = tmp_path / "three_rows.csv"
    csv_path.write_text(
        "ruleset,trad_result\n"
        "\"TESTCASE=MAJORITY\\nrule \\\"A\\\" when Item X changed then end\",SAC\n"
        "\"TESTCASE=TIE\\nrule \\\"B\\\" when Item Y changed then end\",WTC\n"
        "\"TESTCASE=MAJORITY\\nrule \\\"C\\\" when Item Z changed then end\",SAC\n",
        encoding="utf-8",
    )

    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "dataset_path": str(csv_path),
            "ruleset_column": "ruleset",
            "gold_column": "trad_result",
            "max_rows": 3,
        },
    }

    # Reset counter before test
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None

    # 3 rows evaluated
    assert data["metrics"]["rows_evaluated"] == 3
    assert data["metrics"]["accuracy"] == 1.0  # All should match

    # 3 calls (1 per row)
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 3, f"Expected 3 purple calls for 3 rows, got {calls}"


@pytest.mark.asyncio
async def test_green_accuracy_calculation(agent, mock_purple, mutation_data):
    """
    Test that green agent correctly calculates accuracy.
    """
    csv_path = mutation_data / "mutated_dataset.csv"
    csv_path.write_text(
        "ruleset,trad_result\n"
        # Correct (purple returns SAC for MAJORITY)
        "\"TESTCASE=MAJORITY\\nrule \\\"A\\\"\",SAC\n"
        # Correct (purple returns WTC for TIE)
        "\"TESTCASE=TIE\\nrule \\\"B\\\"\",WTC\n"
        # Incorrect (purple returns SAC but gold is WTC)
        "\"TESTCASE=MAJORITY\\nrule \\\"C\\\"\",WTC\n",
        encoding="utf-8",
    )

    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "dataset_path": str(csv_path),
            "max_rows": 3,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None

    # Should have 2 correct out of 3
    assert data["metrics"]["rows_evaluated"] == 3
    assert data["metrics"]["correct"] == 2
    assert abs(data["metrics"]["accuracy"] - 0.6667) < 0.01  # ~66.67%


@pytest.mark.asyncio
async def test_green_label_stats(agent, mock_purple, mutation_data):
    """
    Test that green agent correctly tracks label statistics.
    """
    csv_path = mutation_data / "mutated_dataset.csv"
    csv_path.write_text(
        "ruleset,trad_result\n"
        "\"TESTCASE=MAJORITY\\nrule \\\"A\\\"\",SAC\n"
        "\"TESTCASE=MAJORITY\\nrule \\\"B\\\"\",SAC\n"
        "\"TESTCASE=TIE\\nrule \\\"C\\\"\",WTC\n",
        encoding="utf-8",
    )

    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "dataset_path": str(csv_path),
            "max_rows": 3,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None

    # Check label counts
    assert data["label_stats"]["pred_counts"]["SAC"] == 2
    assert data["label_stats"]["pred_counts"]["WTC"] == 1
    assert data["label_stats"]["pred_correct_counts"]["SAC"] == 2
    assert data["label_stats"]["pred_correct_counts"]["WTC"] == 1


@pytest.mark.asyncio
async def test_green_max_rows_limit(agent, mock_purple, tmp_path):
    """
    Test that green agent respects max_rows config.
    """
    csv_path = tmp_path / "many_rows.csv"
    lines = ["ruleset,trad_result"]
    for i in range(100):
        lines.append(f'"TESTCASE=MAJORITY\\nrule \\\"Rule{i}\\\"",SAC')
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "dataset_path": str(csv_path),
            "max_rows": 10,  # Only process first 10
        },
    }

    # Reset counter
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None

    # Should only process 10 rows
    assert data["metrics"]["rows_evaluated"] == 10

    # Should only make 10 calls
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 10, f"Expected 10 purple calls, got {calls}"


@pytest.mark.asyncio
async def test_green_config_used_tracking(agent, mock_purple, tmp_path):
    """
    Test that green agent tracks config it used.
    """
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("ruleset,trad_result\n\"test\",SAC\n", encoding="utf-8")

    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "dataset_path": str(csv_path),
            "ruleset_column": "ruleset",
            "gold_column": "trad_result",
            "max_rows": 5,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None

    # Check config_used is tracked
    config_used = data["config_used"]
    assert config_used["ruleset_column"] == "ruleset"
    assert config_used["gold_column"] == "trad_result"
    assert config_used["max_rows"] == 5
    assert "allowed_labels" in config_used
    
    # Verify no tie_break in config (that was old voting logic)
    assert "tie_break" not in config_used