import argparse
import asyncio
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Task, InvalidRequestError, UnsupportedOperationError
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils.errors import ServerError

CALL_COUNT = 0

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


def _detect_variant(full_text: str) -> int:
    """
      PROMPT + "\n\n===== INPUT START =====\n" + RULESET + "\n===== INPUT END ====="
    """
    prefix = full_text.split("===== INPUT START =====")[0]
    n = prefix.count('rule "')
    # In the prompt files: ~1 (0-shot), ~13 (1-shot), ~25 (2-shot) before input is appended
    if n < 5:
        return 0
    if n < 20:
        return 1
    return 2


class MockPurpleAgent:
    """Simple agent logic - just the run method"""
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        global CALL_COUNT
        CALL_COUNT += 1

        text = get_message_text(message) or ""
        variant = _detect_variant(text)

        # Choose a deterministic response pattern based on a marker in the input text.
        # The test will include one of these markers inside the ruleset.
        if "TESTCASE=TIE" in text:
            # 0 -> WAC, 1 -> SAC, 2 -> WTC  (all different, tie-break should pick 2-shot => WTC)
            labels = ["WAC", "SAC", "WTC"]
        else:
            # 0 -> SAC, 1 -> SAC, 2 -> WAC  (majority => SAC)
            labels = ["SAC", "SAC", "WAC"]

        label = labels[variant]

        await updater.update_status(TaskState.completed, new_agent_text_message(label))


class MockPurpleExecutor(AgentExecutor):
    """Executor wrapper - handles the execute interface"""
    def __init__(self):
        self.agents: dict[str, MockPurpleAgent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already processed"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        updater = TaskUpdater(event_queue, task.id, context_id)
        
        await updater.start_work()
        try:
            agent = self.agents.get(context_id)
            if not agent:
                agent = MockPurpleAgent()
                self.agents[context_id] = agent
            
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Mock purple task failed: {e}")
            import traceback
            traceback.print_exc()
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context_id, task_id=task.id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


async def debug_calls(request):
    return JSONResponse({"call_count": CALL_COUNT})


async def debug_reset(request):
    global CALL_COUNT
    CALL_COUNT = 0
    return JSONResponse({"ok": True})


def build_app(host: str, port: int, card_url: str | None):
    skill = AgentSkill(
        id="mock_label",
        name="Mock Labeler",
        description="Returns deterministic 3-letter label for testing green agent.",
        tags=["test"],
        examples=["Return SAC"],
    )

    agent_card = AgentCard(
        name="Mock Purple Agent",
        description="Mock purple agent for local tests.",
        url=card_url or f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=MockPurpleExecutor(),
        task_store=InMemoryTaskStore(),
    )

    a2a_server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler).build()

    # Mount A2A server at "/" and add debug endpoints.
    return Starlette(
        routes=[
            Route("/debug/calls", debug_calls, methods=["GET"]),
            Route("/debug/reset", debug_reset, methods=["POST"]),
            Mount("/", app=a2a_server),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    app = build_app(args.host, args.port, args.card_url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()