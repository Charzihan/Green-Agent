import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
sys.path.append(str(Path(__file__).resolve().parent))

from test_agent import send_text_message


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout_s: float = 10.0) -> None:
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
    port = _get_free_port()
    url = f"http://127.0.0.1:{port}"

    script = Path(__file__).parent / "mock_purple_server.py"
    proc = subprocess.Popen(
        [sys.executable, str(script), "--host", "127.0.0.1", "--port", str(port), "--card-url", f"{url}/"],

    )

    try:
        _wait_ready(url)
        # reset counter
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
    Find the DataPart in the task's artifacts (when streaming=False).
    When streaming is disabled, we get the final task state with artifacts attached.
    """
    for event in events:
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event
            
            # Check task.artifacts directly (for non-streaming case)
            if hasattr(task, 'artifacts') and task.artifacts:
                for artifact in task.artifacts:
                    for part in getattr(artifact, 'parts', []) or []:
                        root = getattr(part, 'root', None)
                        if root and hasattr(root, 'data'):
                            return root.data
            
            # Also check update.artifact (for streaming case)
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
async def test_green_majority_vote(agent, mock_purple, tmp_path):
    # Create a tiny dataset so the test is fast and deterministic
    csv_path = tmp_path / "tiny.csv"
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

    assert data["metrics"]["rows_evaluated"] == 1
    assert data["metrics"]["accuracy"] == 1.0

    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 3, f"Expected 3 purple calls for 1 row, got {calls}"


@pytest.mark.asyncio
async def test_green_tie_break_prefers_2shot(agent, mock_purple, tmp_path):
    csv_path = tmp_path / "tiny_tie.csv"
    csv_path.write_text(
        "ruleset,trad_result\n"
        "\"TESTCASE=TIE\\nrule \\\"B\\\" when Item Y changed then end\",WTC\n",
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

    # reset counter for this test
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None, "Green agent did not emit a data artifact"
    assert data["metrics"]["accuracy"] == 1.0

    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 3
