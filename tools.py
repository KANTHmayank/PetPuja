"""
The three "physical" operations from the whiteboard: take_order, cook, serve.

These are plain, deterministic Python functions (not LLM calls) so they're
easy to unit test on their own. Each is also wrapped with @tool so it can be
handed to an LLM directly later, if you want to move to an agentic
(tool-calling) version instead of the hand-wired graph.

Menu data now lives in SQLite (db.py) instead of an in-memory dict.
"""

import math
import random
import re

from langchain_core.tools import tool

import db

# Progressive failure rates for live demo & portfolio showcasing:
# 1st attempt has ~25% chance of transient hiccup to demonstrate LangGraph self-healing.
# On retry, failure drops to ~5% so items recover reliably.
INITIAL_COOK_FAILURE_RATE = 0.25
RETRY_COOK_FAILURE_RATE = 0.05
INITIAL_SERVE_FAILURE_RATE = 0.12
RETRY_SERVE_FAILURE_RATE = 0.03


@tool
def take_order(item: str, qty: int) -> dict:
    """Check whether `item` is on the menu and `qty` units are in stock.
    Supports both portion-level and piece-level pricing (e.g. 5 pcs gulab jamun @ $2.25/pc).
    """
    clean_item = item.lower().strip()
    is_piece_order = False

    # Check if item name specifies (pcs) or (piece)
    if "(pcs)" in clean_item or "(pieces)" in clean_item or "(pc)" in clean_item:
        is_piece_order = True
        clean_item = re.sub(r"\(pcs\)|\(pieces\)|\(pc\)", "", clean_item).strip()

    # If the user passed something like "3 pcs samosa", extract 3, unit, and item name
    match = re.match(
        r"^\s*(\d+)\s*(pcs|pc|pieces|piece|plates|plate|portions|portion|glasses|glass|bottles|bottle|cans|can|bowls|bowl|servings|serving|pints|pint)?\s*(?:of\s+)?(.*)$",
        clean_item,
    )
    if match:
        extracted_qty = int(match.group(1))
        unit = match.group(2)
        if unit in ["pcs", "pc", "pieces", "piece"]:
            is_piece_order = True
        if extracted_qty > 0 and qty == 1:
            qty = extracted_qty
        remainder = match.group(3).strip()
        if remainder:
            clean_item = remainder

    row = db.get_item(clean_item)

    if row is None:
        return {"status": "unavailable", "reason": f"'{item}' is not on the menu"}

    canonical_name = row["name"]
    piece_info = db.get_piece_info(canonical_name)

    # If this is a piece-level order for an item that has pieces per portion:
    if is_piece_order and piece_info["has_pieces"]:
        price = piece_info["price_per_piece"]
        portions_needed = math.ceil(qty / piece_info["pieces_per_portion"])
        if row["stock"] < portions_needed:
            return {
                "status": "unavailable",
                "reason": f"only {row['stock']} portions ({row['stock'] * piece_info['pieces_per_portion']} pcs) of '{canonical_name}' left, {qty} pcs requested",
                "item": f"{canonical_name} (pcs)",
                "qty": qty,
            }
        return {
            "status": "valid",
            "reason": f"{qty} pcs {canonical_name} confirmed (@ ${price:.2f}/pc)",
            "price": price,
            "item": f"{canonical_name} (pcs)",
            "qty": qty,
        }

    # Standard portion-level order
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
def cook(item: str, qty: int, retries: int = 0) -> dict:
    """Simulate cooking `qty` units of `item`. Fails randomly to exercise retries.
    Uses progressive failure: ~25% on first attempt (retries == 0) to demonstrate
    LangGraph self-healing, dropping to ~5% on retries so items recover and cook.
    """
    fail_rate = INITIAL_COOK_FAILURE_RATE if retries == 0 else RETRY_COOK_FAILURE_RATE
    if random.random() < fail_rate:
        reasons = [
            f"Tandoor temperature dropped while making {item}",
            f"Spices needed re-simmering for {item}",
            f"Oil splatter during sautéing {item}",
            f"Handi steam pressure dropped for {item}",
        ]
        return {"status": "failed", "reason": random.choice(reasons)}
    return {"status": "done", "reason": f"{qty}x {item} cooked to perfection"}


@tool
def serve(item: str, qty: int, retries: int = 0) -> dict:
    """Simulate serving `qty` units of `item`. Fails randomly to exercise retries.
    Uses progressive failure: ~12% on first attempt, dropping to ~3% on retry.
    """
    clean_item = item.lower().strip()
    fail_rate = INITIAL_SERVE_FAILURE_RATE if retries == 0 else RETRY_SERVE_FAILURE_RATE
    if random.random() < fail_rate:
        reasons = [
            f"Serving tray unbalanced with {item}",
            f"Table placement delay for {item}",
            f"Fresh coriander garnish adjustment on {item}",
        ]
        return {"status": "failed", "reason": random.choice(reasons)}

    # Convert pieces to portions for inventory decrement
    base_item = re.sub(r"\(pcs\)|\(pieces\)|\(pc\)", "", clean_item).strip()
    piece_info = db.get_piece_info(base_item)
    if piece_info["has_pieces"] and any(p in clean_item for p in ["(pcs)", "(pieces)", "(pc)"]):
        portions = math.ceil(qty / piece_info["pieces_per_portion"])
        db.decrement_stock(base_item, portions)
    else:
        db.decrement_stock(base_item, qty)

    return {"status": "done", "reason": f"{qty}x {item} served hot"}