import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from graph import build_graph
from state import initial_state

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise SystemExit(
        "Set GROQ_API_KEY in your environment or a .env file before running this."
    )


import uuid

from langgraph.types import Command


def main():
    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Restaurant agent ready. Try: 'I'd like 2 burgers'. Ctrl+C to quit.\n")

    while True:
        # Check if graph is paused at an interrupt (confirmation)
        current_state = app.get_state(config)
        is_interrupted = bool(current_state.tasks and any(t.interrupts for t in current_state.tasks))

        if is_interrupted:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            result = app.invoke(Command(resume=user_input), config=config)
        else:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            result = app.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "items": [],
                    "cook_status": "pending",
                    "serve_status": "pending",
                    "cook_retries": 0,
                    "serve_retries": 0,
                    "total_bill": None,
                    "order_status": None,
                    "error": None,
                },
                config=config,
            )

        # Check if the execution ended at an interrupt
        new_state = app.get_state(config)
        if new_state.tasks and any(t.interrupts for t in new_state.tasks):
            interrupt_val = new_state.tasks[0].interrupts[0].value
            print(f"Agent:\n{interrupt_val}")
        elif result.get("messages"):
            print(f"Agent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()