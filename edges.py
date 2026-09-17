"""
Routing functions: each reads `state` and returns a string key that
add_conditional_edges maps to the next node. This is the "if-else / switch"
logic from the whiteboard, kept out of the nodes themselves so the branching
is easy to scan in one place.
"""

from state import RestaurantState
from nodes import MAX_COOK_RETRIES, MAX_SERVE_RETRIES


def route_after_router(state: RestaurantState) -> str:
    # No items means the router already composed the final reply itself
    # (a menu question or an off-topic message) — go straight to END so
    # respond_node doesn't overwrite that message with its own.
    return "take_order" if state.get("items") else "end"


def route_after_order(state: RestaurantState) -> str:
    all_valid = bool(state["items"]) and all(l["menu_status"] == "valid" for l in state["items"])
    return "confirm_order" if all_valid else "respond"


def route_after_confirm(state: RestaurantState) -> str:
    if state.get("order_status") == "confirmed":
        return "cook"
    return "end"


def route_after_cook(state: RestaurantState) -> str:
    valid_items = [l for l in state["items"] if l["menu_status"] == "valid"]
    pending = [l for l in valid_items if l["cook_status"] != "done"]

    if not pending:
        return "serve"
    # If any pending item still has retries remaining, retry cooking
    if any(l["cook_retries"] < MAX_COOK_RETRIES for l in pending):
        return "cook"
    # If cook retries are exhausted for failed items, STILL serve whatever was cooked!
    cooked = [l for l in valid_items if l["cook_status"] == "done"]
    if cooked:
        return "serve"
    return "respond"  # No items could be cooked at all


def route_after_serve(state: RestaurantState) -> str:
    cooked_items = [l for l in state["items"] if l["cook_status"] == "done"]
    pending = [l for l in cooked_items if l["serve_status"] != "done"]

    if not pending:
        return "respond"
    # If any pending item still has retries remaining, retry serving
    if any(l["serve_retries"] < MAX_SERVE_RETRIES for l in pending):
        return "serve"
    return "respond"  # give up, explain the failure