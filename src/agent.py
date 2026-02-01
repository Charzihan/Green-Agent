from __future__ import annotations

import asyncio
from typing import Any, Optional
from pathlib import Path
import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import time

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any] = {}


ALLOWED_LABELS = {"WAC", "SAC", "WTC", "STC", "WCC", "SCC"}


class RowStatus(str, Enum):
    """Status of individual row processing - like a grade for each exam."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    PARSE_FAILED = "parse_failed"
    SKIPPED = "skipped"  # Skipped due to circuit breaker


@dataclass
class RowResult:
    """Result of processing a single row."""
    row_index: int
    gold: str
    pred: Optional[str]
    status: RowStatus
    is_correct: bool
    response_preview: str = ""
    error_message: str = ""
    duration_ms: int = 0


@dataclass 
class AssessmentState:
    """
    Tracks assessment progress - like a scoreboard that persists even if the game is interrupted.
    
    Key insight: We always have SOME results to report, even if we fail partway through.
    """
    total_attempted: int = 0
    total_successful: int = 0
    correct: int = 0
    per_label: Counter = field(default_factory=Counter)
    per_label_correct: Counter = field(default_factory=Counter)
    row_results: list[RowResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    # Circuit breaker state
    consecutive_failures: int = 0
    total_failures: int = 0
    circuit_broken: bool = False
    
    # Filtering state
    rows_skipped_by_filter: int = 0
    rows_scanned: int = 0  # Total rows looked at (including filtered)
    total_matching_in_dataset: int = 0  # Total rows matching filter in entire dataset
    
    # Timing
    start_time: float = field(default_factory=time.time)
    
    def record_success(self, result: RowResult):
        """Record a successful row - like marking an exam as graded."""
        self.total_attempted += 1
        self.total_successful += 1
        self.consecutive_failures = 0  # Reset streak on success
        
        if result.pred:
            self.per_label[result.pred] += 1
        
        if result.is_correct:
            self.correct += 1
            if result.pred:
                self.per_label_correct[result.pred] += 1
        
        self._add_result(result)
    
    def record_failure(self, result: RowResult):
        """Record a failed row - but don't lose it!"""
        self.total_attempted += 1
        self.consecutive_failures += 1
        self.total_failures += 1
        
        if result.error_message:
            self.errors.append(f"Row {result.row_index}: {result.error_message}")
        
        self._add_result(result)
    
    def record_filtered(self):
        """Record a row that was skipped due to RIT filter."""
        self.rows_skipped_by_filter += 1
    
    def _add_result(self, result: RowResult):
        """Keep results (with a reasonable limit to avoid huge payloads)."""
        if len(self.row_results) < 100:  # Keep first 100 for debugging
            self.row_results.append(result)
    
    @property
    def accuracy(self) -> float:
        """Accuracy over successfully processed rows."""
        if self.total_successful == 0:
            return 0.0
        return self.correct / self.total_successful
    
    @property
    def success_rate(self) -> float:
        """What percentage of API calls succeeded."""
        if self.total_attempted == 0:
            return 0.0
        return self.total_successful / self.total_attempted
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    def to_result_dict(self, config_used: dict, early_termination_reason: str = "") -> dict:
        """
        Convert to final result - THIS ALWAYS PRODUCES VALID OUTPUT.
        
        Even if we processed 0 rows, we return a valid structure.
        """
        sample_results = [
            {
                "row_index": r.row_index,
                "gold": r.gold,
                "pred": r.pred,
                "status": r.status.value,
                "is_correct": r.is_correct,
                "response_preview": r.response_preview[:200] if r.response_preview else "",
                "error": r.error_message,
                "duration_ms": r.duration_ms,
            }
            for r in self.row_results[:50]  # First 50 for artifact
        ]
        
        return {
            "metrics": {
                "rows_attempted": self.total_attempted,
                "rows_successful": self.total_successful,
                "rows_skipped_by_filter": self.rows_skipped_by_filter,
                "rows_scanned": self.rows_scanned,
                "total_matching_in_dataset": self.total_matching_in_dataset,
                "correct": self.correct,
                "accuracy": round(self.accuracy, 4),
                "success_rate": round(self.success_rate, 4),
                "total_failures": self.total_failures,
                "elapsed_seconds": round(self.elapsed_seconds, 2),
            },
            "label_stats": {
                "pred_counts": dict(self.per_label),
                "pred_correct_counts": dict(self.per_label_correct),
            },
            "samples": sample_results,
            "errors": self.errors[:20],  # First 20 errors
            "config_used": config_used,
            "early_termination_reason": early_termination_reason,
        }


def _extract_label(model_output: str) -> Optional[str]:
    """
    Robust parsing: accept exactly one of the allowed 3-letter labels.
    """
    text = (model_output or "").upper()
    hits = [lab for lab in ALLOWED_LABELS if re.search(rf"\b{lab}\b", text)]
    if len(hits) == 1:
        return hits[0]
    if text.strip() in ALLOWED_LABELS:
        return text.strip()
    return None


def _find_repo_root() -> Path:
    """Find the repository root by looking for marker files/directories."""
    current = Path(__file__).resolve().parent
    
    if current.name == "src":
        repo_root = current.parent
    else:
        repo_root = current
    
    if (repo_root / "mutation_data").exists() and (repo_root / "prompts").exists():
        return repo_root
    
    repo_root = repo_root.parent
    if (repo_root / "mutation_data").exists() and (repo_root / "prompts").exists():
        return repo_root
    
    return Path(__file__).resolve().parents[1]


class Agent:
    """
    Green agent for RIT classification benchmark.
    
    Robustness features (think of it like a well-designed assembly line):
    1. Health check - verify purple agent is alive before starting
    2. Per-row timeout - don't let one slow response block everything
    3. Retry logic - transient failures get a second chance
    4. Circuit breaker - stop early if purple agent is clearly dead
    5. Partial results - ALWAYS return what we have, even on failure
    """
    
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = []
    
    # Robustness defaults (can be overridden via config)
    DEFAULT_ROW_TIMEOUT = 60  # seconds per purple agent call
    DEFAULT_MAX_RETRIES = 2  # retry failed calls this many times
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures before stopping
    DEFAULT_HEALTH_CHECK_TIMEOUT = 10  # seconds to verify purple is alive

    def __init__(self):
        self.messenger = Messenger()
        
        # Path resolution
        self.repo_root = _find_repo_root()
        self.default_dataset_path = self.repo_root / "mutation_data" / "mutated_dataset.csv"
        self.default_prompt_path = self.repo_root / "prompts" / "prompt_3letter_2shot_NOmultiple.txt"

        print(f"DEBUG: Repo root: {self.repo_root}")
        print(f"DEBUG: Dataset exists: {self.default_dataset_path.exists()}")

        # Load prompt
        if self.default_prompt_path.exists():
            self.prompt = self.default_prompt_path.read_text(encoding="utf-8")
        else:
            print(f"WARNING: Prompt file not found at {self.default_prompt_path}")
            self.prompt = "Classify the following openHAB rules for RIT threats:"

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        return True, "ok"

    async def _health_check(self, purple_url: str, timeout: int) -> tuple[bool, str]:
        """
        Verify purple agent is alive before starting the benchmark.
        
        Like checking if a student is present before handing them an exam.
        """
        try:
            # Send a simple ping message
            response = await asyncio.wait_for(
                self.messenger.talk_to_agent(
                    "Health check - please respond with OK",
                    purple_url,
                    new_conversation=True,
                    timeout=timeout,
                ),
                timeout=timeout + 5  # Extra buffer for asyncio timeout
            )
            return True, f"Health check passed: {response[:100]}"
        except asyncio.TimeoutError:
            return False, f"Health check timed out after {timeout}s"
        except Exception as e:
            return False, f"Health check failed: {type(e).__name__}: {e}"

    async def _ask_purple_with_retry(
        self,
        purple_url: str,
        ruleset_text: str,
        row_index: int,
        timeout: int,
        max_retries: int,
    ) -> RowResult:
        """
        Ask purple agent with timeout and retry logic.
        
        Like a teacher giving a student multiple chances to answer,
        but with a time limit for each attempt.
        """
        gold = ""  # Will be set by caller
        
        # A2A payload format
        payload = (
            f"{self.prompt}\n\n"
            "===== RULES START =====\n"
            f"{ruleset_text}\n"
            "===== RULES END =====\n"
        )
        
        last_error = ""
        
        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                # Use asyncio.wait_for for timeout (more reliable than httpx timeout alone)
                response = await asyncio.wait_for(
                    self.messenger.talk_to_agent(
                        payload,
                        purple_url,
                        timeout=timeout,
                    ),
                    timeout=timeout + 5  # Small buffer
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                pred = _extract_label(response)
                
                # Success! (even if we couldn't parse a label)
                status = RowStatus.SUCCESS if pred else RowStatus.PARSE_FAILED
                
                return RowResult(
                    row_index=row_index,
                    gold="",  # Set by caller
                    pred=pred,
                    status=status,
                    is_correct=False,  # Set by caller
                    response_preview=response[:300] if response else "",
                    duration_ms=duration_ms,
                )
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s (attempt {attempt + 1}/{max_retries + 1})"
                print(f"DEBUG: Row {row_index} - {last_error}")
                
            except Exception as e:
                last_error = f"{type(e).__name__}: {e} (attempt {attempt + 1}/{max_retries + 1})"
                print(f"DEBUG: Row {row_index} - {last_error}")
            
            # Brief pause before retry
            if attempt < max_retries:
                await asyncio.sleep(1)
        
        # All retries exhausted
        duration_ms = int((time.time() - start_time) * 1000)
        is_timeout = "Timeout" in last_error or "timed out" in last_error.lower()
        
        return RowResult(
            row_index=row_index,
            gold="",
            pred=None,
            status=RowStatus.TIMEOUT if is_timeout else RowStatus.ERROR,
            is_correct=False,
            error_message=last_error,
            duration_ms=duration_ms,
        )

    async def _emit_partial_results(
        self,
        updater: TaskUpdater,
        state: AssessmentState,
        config_used: dict,
        reason: str,
        is_failure: bool = True,
    ):
        """
        Emit results even on failure - the show must go on!
        
        This ensures we NEVER lose completed work.
        """
        result = state.to_result_dict(config_used, early_termination_reason=reason)
        
        # Build summary with filter info
        rit_filter = config_used.get("rit_filter")
        filter_info = f" [filter: {rit_filter}]" if rit_filter else ""
        
        summary = (
            f"{'PARTIAL RESULTS' if is_failure else 'COMPLETED'}{filter_info}: "
            f"Accuracy {state.accuracy:.2%} ({state.correct}/{state.total_successful}) "
            f"| {state.total_attempted} rows evaluated "
            f"| {state.total_failures} failures"
        )
        
        if rit_filter:
            # Show how many total matching rows exist in the dataset
            summary += f" | Total {rit_filter} rows in dataset: {state.total_matching_in_dataset}"
        
        if reason:
            summary += f" | Note: {reason}"
        
        print(f"DEBUG: Emitting results - {summary}")
        
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(kind="text", text=summary)),
                Part(root=DataPart(kind="data", data=result)),
            ],
            name="Result",
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Main assessment loop with full robustness.
        """
        # Reset messenger for isolation
        self.messenger.reset()
        
        # Parse request
        input_text = get_message_text(message)
        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        purple_url = str(request.participants["agent"])
        
        # Extract config with defaults
        cfg = request.config or {}
        dataset_path = Path(cfg.get("dataset_path", self.default_dataset_path))
        ruleset_col = cfg.get("ruleset_column", "ruleset")
        gold_col = cfg.get("gold_column", "trad_result")
        max_rows = int(cfg.get("max_rows", 10000))
        
        # RIT type filter - only evaluate rows with this specific label
        # If None or empty, evaluate all RIT types
        rit_filter = cfg.get("rit_filter", None) or cfg.get("filter_rit", None)
        if rit_filter:
            rit_filter = rit_filter.strip().upper()
            if rit_filter not in ALLOWED_LABELS:
                await updater.failed(
                    new_agent_text_message(
                        f"Invalid rit_filter '{rit_filter}'. Must be one of: {sorted(ALLOWED_LABELS)}"
                    )
                )
                return
        
        # Robustness config
        row_timeout = int(cfg.get("row_timeout", self.DEFAULT_ROW_TIMEOUT))
        max_retries = int(cfg.get("max_retries", self.DEFAULT_MAX_RETRIES))
        circuit_threshold = int(cfg.get("circuit_breaker_threshold", self.DEFAULT_CIRCUIT_BREAKER_THRESHOLD))
        skip_health_check = cfg.get("skip_health_check", False)
        
        config_used = {
            "dataset_path": str(dataset_path),
            "ruleset_column": ruleset_col,
            "gold_column": gold_col,
            "max_rows": max_rows,
            "rit_filter": rit_filter,  # None means all types
            "row_timeout": row_timeout,
            "max_retries": max_retries,
            "circuit_breaker_threshold": circuit_threshold,
            "allowed_labels": sorted(ALLOWED_LABELS),
        }
        
        # Initialize state tracker
        state = AssessmentState()
        
        # Verify dataset exists
        if not dataset_path.exists():
            await updater.failed(
                new_agent_text_message(f"Dataset not found: {dataset_path}")
            )
            return

        # === HEALTH CHECK ===
        if not skip_health_check:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Checking purple agent health at {purple_url}..."),
            )
            
            healthy, health_msg = await self._health_check(
                purple_url, 
                self.DEFAULT_HEALTH_CHECK_TIMEOUT
            )
            
            if not healthy:
                await updater.failed(
                    new_agent_text_message(f"Purple agent not responding: {health_msg}")
                )
                return
            
            print(f"DEBUG: {health_msg}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting benchmark: {max_rows} rows, {row_timeout}s timeout per row"
                + (f", filtering for RIT type: {rit_filter}" if rit_filter else ", all RIT types")
            ),
        )

        # === MAIN PROCESSING LOOP ===
        termination_reason = ""
        
        try:
            with dataset_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    # Track total rows scanned
                    state.rows_scanned = i + 1
                    
                    ruleset_text = row.get(ruleset_col, "")
                    gold = (row.get(gold_col, "") or "").strip().upper()
                    
                    # === RIT FILTER CHECK ===
                    # Skip rows that don't match the filter (if filter is set)
                    if rit_filter:
                        if gold == rit_filter:
                            state.total_matching_in_dataset += 1
                        else:
                            state.record_filtered()
                            continue
                    
                    # Check if we've processed enough matching rows
                    if state.total_attempted >= max_rows:
                        # If filtering, keep scanning to count total matching rows
                        if rit_filter:
                            continue  # Keep counting but don't process
                        else:
                            break  # No filter, just stop
                    
                    # === CIRCUIT BREAKER CHECK ===
                    if state.consecutive_failures >= circuit_threshold:
                        termination_reason = (
                            f"Circuit breaker triggered: {state.consecutive_failures} "
                            f"consecutive failures"
                        )
                        state.circuit_broken = True
                        print(f"DEBUG: {termination_reason}")
                        # If filtering, keep scanning to count total
                        if rit_filter:
                            continue
                        else:
                            break
                    
                    # === PROCESS SINGLE ROW (with retries) ===
                    result = await self._ask_purple_with_retry(
                        purple_url=purple_url,
                        ruleset_text=ruleset_text,
                        row_index=i,
                        timeout=row_timeout,
                        max_retries=max_retries,
                    )
                    
                    # Fill in gold label and correctness
                    result.gold = gold
                    result.is_correct = (result.pred == gold) if result.pred else False
                    
                    # Record result (success or failure - we keep everything!)
                    if result.status in (RowStatus.SUCCESS, RowStatus.PARSE_FAILED):
                        state.record_success(result)
                    else:
                        state.record_failure(result)
                    
                    # Progress update every 5 rows
                    if state.total_attempted % 5 == 0:
                        filter_info = f" (filter: {rit_filter})" if rit_filter else ""
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                f"Progress: {state.total_attempted}/{max_rows} rows{filter_info} "
                                f"| Accuracy: {state.accuracy:.1%} "
                                f"| Failures: {state.total_failures}"
                                + (f" | Skipped: {state.rows_skipped_by_filter}" if rit_filter else "")
                            ),
                        )
        
        except Exception as e:
            # Unexpected error - but we STILL emit partial results!
            termination_reason = f"Unexpected error: {type(e).__name__}: {e}"
            print(f"DEBUG: {termination_reason}")
            import traceback
            traceback.print_exc()
        
        # Check if filter resulted in fewer rows than requested
        if rit_filter:
            if state.total_attempted == 0 and state.total_matching_in_dataset == 0:
                termination_reason = (
                    f"No rows found with RIT type '{rit_filter}' in dataset "
                    f"({state.rows_scanned} rows scanned, all had different RIT types)"
                )
            elif state.total_attempted < max_rows:
                # We found some but not enough - this is informational, not an error
                if not termination_reason:  # Don't override error messages
                    termination_reason = (
                        f"Dataset contains only {state.total_matching_in_dataset} rows "
                        f"with RIT type '{rit_filter}' (requested {max_rows})"
                    )
        
        # === ALWAYS EMIT RESULTS ===
        # Determine if this is actually a failure or just informational
        # "Fewer matching rows than requested" is informational, not a failure
        is_actual_failure = state.circuit_broken or (
            termination_reason and 
            "Unexpected error" in termination_reason
        )
        
        await self._emit_partial_results(
            updater=updater,
            state=state,
            config_used=config_used,
            reason=termination_reason,
            is_failure=is_actual_failure,
        )
        
        # Set final status
        if is_actual_failure:
            # Even on "failure", we completed with partial results
            # Use 'completed' if we have meaningful results, 'failed' only if we got nothing
            if state.total_successful > 0:
                await updater.update_status(
                    TaskState.completed,
                    new_agent_text_message(
                        f"Completed with issues: {termination_reason}"
                    ),
                )
            else:
                await updater.failed(
                    new_agent_text_message(f"Failed: {termination_reason}")
                )
        else:
            await updater.update_status(
                TaskState.completed,
                new_agent_text_message("Assessment completed successfully.")
            )