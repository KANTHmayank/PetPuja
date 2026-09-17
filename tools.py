"""
The three "physical" operations from the whiteboard: take_order, cook, serve.

These are plain, deterministic Python functions (not LLM calls) so they're
easy to unit test on their own. Each is also wrapped with @tool so it can be
handed to an LLM directly later, if you want to move to an agentic
(tool-calling) version instead of the hand-wired graph.

Menu data now lives in SQLite (db.py) instead of an in-memory dict.
"""

import random
import re

from langchain_core.tools import tool

import db

# Realistic failure rates for live demo (exercises resilience without constantly ruining orders)
COOK_FAILURE_RATE = 0.08
SERVE_FAILURE_RATE = 0.05


@tool
def take_order(item: str, qty: int) -> dict:
    """Check whether `item` is on the menu and `qty` units are in stock."""
    clean_item = item.lower().strip()
    # If the user passed something like "3 pcs samosa" and qty is 1, extract 3 and clean item name
    match = re.match(r"^\s*(\d+)\s*(?:pcs|pc|pieces|piece|plates|plate|portions|portion|glasses|glass|bottles|bottle|cans|can|bowls|bowl|servings|serving|pints|pint)?\s*(?:of\s+)?(.*)$", clean_item)
    if match:
        extracted_qty = int(match.group(1))
        if extracted_qty > 0 and qty == 1:
            qty = extracted_qty
        remainder = match.group(2).strip()
        if remainder:
            clean_item = remainder

    row = db.get_item(clean_item)

    if row is None:
        return {"status": "unavailable", "reason": f"'{item}' is not on the menu"}

    canonical_name = row["name"]
    if row["stock"] < qty:
        return {
            "status": "unavailable",
            "reason": f"only {row['stock']} '{canonical_name}' left, {qty} requested",
            "item": canonical_name,
            "qty": qty,
        }
    return {
        "status": "valid",
        "reason": f"{qty}x {canonical_name} confirmed",
        "price": row["price"],
        "item": canonical_name,
        "qty": qty,
    }


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