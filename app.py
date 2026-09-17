import os
import uuid
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import build_graph
import db

app = FastAPI(title="PetPuja - AI Bistro API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile agent graph with MemorySaver checkpointer
agent_app = build_graph()


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ConfirmRequest(BaseModel):
    action: str  # "yes" or "no"
    thread_id: str


@app.get("/api/menu")
def get_menu():
    """Return all menu items with real-time stock and prices."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT id, name, category, price, stock, description FROM menu_items ORDER BY category, name"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "price": r["price"],
            "stock": r["stock"],
            "description": r["description"],
        }
        for r in rows
    ]


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Send customer message to PetPuja's AI Maitre D'."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Check if graph is already waiting at an interrupt
        current_state = agent_app.get_state(config)
        if current_state.tasks and any(t.interrupts for t in current_state.tasks):
            # Graph is paused at an interrupt, user input is treated as the resume response
            result = agent_app.invoke(Command(resume=req.message), config=config)
        else:
            result = agent_app.invoke(
                {
                    "messages": [HumanMessage(content=req.message)],
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

        new_state = agent_app.get_state(config)
        if new_state.tasks and any(t.interrupts for t in new_state.tasks):
            interrupt_val = new_state.tasks[0].interrupts[0].value
            return {
                "status": "interrupted",
                "prompt": interrupt_val,
                "thread_id": thread_id,
                "state": {
                    "items": new_state.values.get("items", []),
                    "total_bill": new_state.values.get("total_bill"),
                    "order_status": new_state.values.get("order_status"),
                },
            }

        last_message = result.get("messages", [])[-1].content if result.get("messages") else ""
        return {
            "status": "completed",
            "reply": last_message,
            "thread_id": thread_id,
            "state": {
                "items": new_state.values.get("items", []),
                "cook_status": new_state.values.get("cook_status"),
                "serve_status": new_state.values.get("serve_status"),
                "total_bill": new_state.values.get("total_bill"),
                "order_status": new_state.values.get("order_status"),
                "error": new_state.values.get("error"),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/confirm")
def confirm_order(req: ConfirmRequest):
    """Resume the interrupted graph with confirmation action."""
    config = {"configurable": {"thread_id": req.thread_id}}
    try:
        current_state = agent_app.get_state(config)
        if not (current_state.tasks and any(t.interrupts for t in current_state.tasks)):
            return {
                "status": "not_interrupted",
                "message": "No active order awaiting confirmation.",
                "thread_id": req.thread_id,
            }

        result = agent_app.invoke(Command(resume=req.action), config=config)
        new_state = agent_app.get_state(config)
        last_message = result.get("messages", [])[-1].content if result.get("messages") else ""

        return {
            "status": "completed",
            "reply": last_message,
            "thread_id": req.thread_id,
            "state": {
                "items": new_state.values.get("items", []),
                "cook_status": new_state.values.get("cook_status"),
                "serve_status": new_state.values.get("serve_status"),
                "total_bill": new_state.values.get("total_bill"),
                "order_status": new_state.values.get("order_status"),
                "error": new_state.values.get("error"),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
def reset_database():
    """Reset the SQLite menu database and reseed."""
    db.init_db(reset=True)
    return {"status": "ok", "message": "Menu database re-seeded successfully."}


# Mount static directory for frontend
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
