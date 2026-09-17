"""
The three "physical" operations from the whiteboard: take_order, cook, serve.

These are plain, deterministic Python functions (not LLM calls) so they're
easy to unit test on their own. Each is also wrapped with @tool so it can be
handed to an LLM directly later, if you want to move to an agentic
(tool-calling) version instead of the hand-wired graph.

Menu data now lives in SQLite (db.py) instead of an in-memory dict.
"""

import random

from langchain_core.tools import tool

import db

# Simulated failure rates so the retry loops actually get exercised.
COOK_FAILURE_RATE = 0.35
SERVE_FAILURE_RATE = 0.25


@tool
def take_order(item: str, qty: int) -> dict:
    """Check whether `item` is on the menu and `qty` units are in stock."""
    item = item.lower().strip()
    row = db.get_item(item)

    if row is None:
        return {"status": "unavailable", "reason": f"'{item}' is not on the menu"}
    if row["stock"] < qty:
        return {
            "status": "unavailable",
            "reason": f"only {row['stock']} '{item}' left, {qty} requested",
        }
    return {"status": "valid", "reason": f"{qty}x {item} confirmed", "price": row["price"]}


@tool
def cook(item: str, qty: int) -> dict:
    """Simulate cooking `qty` units of `item`. Fails randomly to exercise retries."""
    if random.random() < COOK_FAILURE_RATE:
        return {"status": "failed", "reason": f"kitchen issue while cooking {item}"}
    return {"status": "done", "reason": f"{qty}x {item} cooked"}


@tool
def serve(item: str, qty: int) -> dict:
    """Simulate serving `qty` units of `item`. Fails randomly to exercise retries.

    On success, decrements DB stock — this is what actually consumes
    inventory, since cooking can still fail after take_order confirms
    availability. Note: because cook/serve currently retry the whole
    batch (see nodes.py), an item that already succeeded once inside a
    batch that later fails and retries will get decremented again on the
    retry — a known simplification, worth fixing with per-item retry
    tracking before this becomes anything real.
    """
    item = item.lower().strip()
    if random.random() < SERVE_FAILURE_RATE:
        return {"status": "failed", "reason": f"serving mishap with {item}"}
    db.decrement_stock(item, qty)
    return {"status": "done", "reason": f"{qty}x {item} served"}