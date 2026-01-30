from __future__ import annotations

from typing import Any, Optional
from pathlib import Path
import csv
import re
from collections import Counter

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


ALLOWED_LABELS = {"WAC", "SAC", "WTC", "STC", "WCC", "SCC"}


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_label(model_output: str) -> Optional[str]:
    """
    Robust parsing: accept exactly one of the allowed 3-letter labels
    even if the model returns extra text.
    """
    text = (model_output or "").upper()
    hits = [lab for lab in ALLOWED_LABELS if re.search(rf"\b{lab}\b", text)]
    if len(hits) == 1:
        return hits[0]
    # If model returned exactly the label without boundaries (rare), fallback:
    if text.strip() in ALLOWED_LABELS:
        return text.strip()
    return None


def _vote(labels: list[Optional[str]]) -> tuple[Optional[str], dict[str, Any]]:
    """
    Majority vote.
    Deterministic tie-break for 1–1–1:
      prefer 2-shot, then 1-shot, then 0-shot (i.e., last -> first).
    """
    clean = [x for x in labels if x in ALLOWED_LABELS]
    meta: dict[str, Any] = {"raw_labels": labels, "valid_labels": clean}

    if not clean:
        meta["decision"] = "no_valid_votes"
        return None, meta

    counts = Counter(clean)
    top_label, top_n = counts.most_common(1)[0]
    if top_n >= 2:
        meta["decision"] = "majority"
        meta["counts"] = dict(counts)
        return top_label, meta

    # 1–1–1 (or all distinct) -> deterministic tie-break: 2-shot > 1-shot > 0-shot
    for preferred in [labels[2], labels[1], labels[0]]:
        if preferred in ALLOWED_LABELS:
            meta["decision"] = "tie_break_prefer_more_shots"
            meta["counts"] = dict(counts)
            return preferred, meta

    meta["decision"] = "tie_break_failed"
    meta["counts"] = dict(counts)
    return None, meta


class Agent:
    # Single purple agent role
    required_roles: list[str] = ["agent"]

    # Keep config flexible; you can add required keys later if you want
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()

        # Package these files inside your green agent repo image.
        # Example layout:
        #   src/agent.py
        #   data/mutated_dataset.csv
        #   prompts/prompt_3letter_0shot_NOmultiple.txt
        #   prompts/prompt_3letter_1shot_NOmultiple.txt
        #   prompts/prompt_3letter_2shot_NOmultiple.txt
        self.repo_root = Path(__file__).resolve().parents[1]
        self.default_dataset_path = self.repo_root / "mutation_data" / "mutated_dataset.csv"
        self.default_prompt_paths = [
            self.repo_root / "prompts" / "prompt_3letter_0shot_NOmultiple.txt",
            self.repo_root / "prompts" / "prompt_3letter_1shot_NOmultiple.txt",
            self.repo_root / "prompts" / "prompt_3letter_2shot_NOmultiple.txt",
        ]

        # Load prompts at startup
        self.prompts = [p.read_text(encoding="utf-8") for p in self.default_prompt_paths]

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def _ask_purple(self, purple_url: str, prompt: str, ruleset_text: str) -> str:
        """
        A2A call to the purple agent.
        """
        payload = (
            f"{prompt}\n\n"
            "===== INPUT START =====\n"
            f"{ruleset_text}\n"
            "===== INPUT END =====\n"
        )
        response: str = await self.messenger.talk_to_agent(payload, purple_url)  # ← Just pass the string!
        return response

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        purple_url = str(request.participants["agent"])

        # Config (optional overrides)
        cfg = request.config or {}
        dataset_path = Path(cfg.get("dataset_path", self.default_dataset_path))
        ruleset_col = cfg.get("ruleset_column", "ruleset")
        gold_col = cfg.get("gold_column", "trad_result")
        max_rows = int(cfg.get("max_rows", 50))  # IMPORTANT: 3 calls per row can be expensive

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Running benchmark: dataset={dataset_path.name}, max_rows={max_rows}"
            ),
        )

        total = 0
        correct = 0
        per_label = Counter()
        per_label_correct = Counter()

        # Optional: keep a small sample of row-level details for debugging
        row_samples: list[dict[str, Any]] = []

        with dataset_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break

                ruleset_text = row.get(ruleset_col, "")
                gold = (row.get(gold_col, "") or "").strip().upper()

                # Never send gold/trad_result to purple!
                outputs = []
                labels = []
                for prompt in self.prompts:
                    out = await self._ask_purple(purple_url, prompt, ruleset_text)
                    outputs.append(out)
                    labels.append(_extract_label(out))

                pred, meta = _vote(labels)

                total += 1
                if pred:
                    per_label[pred] += 1
                if pred == gold:
                    correct += 1
                    if pred:
                        per_label_correct[pred] += 1

                # Keep only a few sample rows to avoid huge artifacts
                if len(row_samples) < 20:
                    row_samples.append(
                        {
                            "row_index": i,
                            "gold": gold,
                            "pred": pred,
                            "votes": labels,
                            "decision": meta.get("decision"),
                            # store short snippets only
                            "purple_raw_outputs_preview": [o[:200] for o in outputs],
                        }
                    )

                if total % 10 == 0:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(f"Progress: {total} rows evaluated..."),
                    )

        accuracy = (correct / total) if total else 0.0

        result = {
            "metrics": {
                "rows_evaluated": total,
                "correct": correct,
                "accuracy": accuracy,
            },
            "label_stats": {
                "pred_counts": dict(per_label),
                "pred_correct_counts": dict(per_label_correct),
            },
            "samples": row_samples,
            "config_used": {
                "dataset_path": str(dataset_path),
                "ruleset_column": ruleset_col,
                "gold_column": gold_col,
                "max_rows": max_rows,
                "allowed_labels": sorted(ALLOWED_LABELS),
                "tie_break": "prefer 2-shot > 1-shot > 0-shot",
            },
        }

        print(f"DEBUG: About to create artifact. Accuracy: {accuracy}, Total: {total}")

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(kind="text", text=f"Accuracy: {accuracy:.4f} ({correct}/{total})")),
                Part(root=DataPart(kind="data", data=result)),
            ],
            name="Result",
        )

        print(f"DEBUG: Artifact created successfully")
        print(f"DEBUG: Terminal state reached: {updater._terminal_state_reached}")

        await updater.update_status(
            TaskState.completed, new_agent_text_message("Done.")
        )
