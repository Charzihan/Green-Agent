"""
Updated Mock Purple Agent for Refactored Green Agent

This mock purple agent simulates a real purple agent that:
1. Receives a prompt + ruleset from the green agent
2. Returns a single 3-letter label
3. Handles its own voting logic internally (if any)

For testing purposes, it returns deterministic labels based on markers in the ruleset.
"""

import argparse
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, InvalidRequestError, UnsupportedOperationError
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils.errors import ServerError

# Global counter for debugging
CALL_COUNT = 0

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class MockPurpleAgent:
    """
    Mock purple agent that returns deterministic labels for testing.
    
    Now expects to be called ONCE per evaluation (green agent no longer does voting).
    Purple agent would handle its own 0/1/2-shot prompting and voting internally.
    """
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        global CALL_COUNT
        CALL_COUNT += 1

        text = get_message_text(message) or ""

        # Detect test case from markers in the ruleset
        # These markers are embedded in the test CSV files
        if "TESTCASE=MAJORITY" in text:
            # Simulate: purple internally runs 3 variants, gets SAC, SAC, WAC → majority SAC
            label = "SAC"
        elif "TESTCASE=TIE" in text:
            # Simulate: purple internally runs 3 variants, gets WAC, SAC, WTC → tie-break WTC (2-shot wins)
            label = "WTC"
        else:
            # Default fallback for unknown test cases
            label = "SAC"

        await updater.update_status(TaskState.completed, new_agent_text_message(label))


class MockPurpleExecutor(AgentExecutor):
    """Executor wrapper for A2A protocol compliance"""
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


# Debug endpoints
async def debug_calls(request):
    """Get number of times purple agent was called"""
    return JSONResponse({"call_count": CALL_COUNT})


async def debug_reset(request):
    """Reset call counter"""
    global CALL_COUNT
    CALL_COUNT = 0
    return JSONResponse({"ok": True})


def build_app(host: str, port: int, card_url: str | None):
    skill = AgentSkill(
        id="mock_rit_classifier",
        name="Mock RIT Classifier",
        description="Returns deterministic 3-letter RIT label for testing. Simulates internal voting logic.",
        tags=["test", "RIT", "classification"],
        examples=[
            "Classify this openHAB ruleset",
            "Detect RIT threats in these rules"
        ],
    )

    agent_card = AgentCard(
        name="Mock Purple Agent (Refactored)",
        description="Mock purple agent that simulates voting logic internally. Returns single label per request.",
        url=card_url or f"http://{host}:{port}/",
        version="2.0.0",  # Bumped to indicate new single-call interface
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

    # Mount A2A server at "/" and add debug endpoints
    return Starlette(
        routes=[
            Route("/debug/calls", debug_calls, methods=["GET"]),
            Route("/debug/reset", debug_reset, methods=["POST"]),
            Mount("/", app=a2a_server),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Mock Purple Agent for testing Green Agent")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9101, help="Port to bind")
    parser.add_argument("--card-url", type=str, default=None, help="URL for agent card")
    args = parser.parse_args()

    print(f"Starting Mock Purple Agent on {args.host}:{args.port}")
    print(f"Debug endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/debug/calls")
    print(f"  POST http://{args.host}:{args.port}/debug/reset")

    app = build_app(args.host, args.port, args.card_url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()