"""
Shared state for the restaurant agent graph.

Every node reads from and writes to this single object. LangGraph merges
each node's returned dict into the running state, so nodes only need to
return the keys they actually changed.
"""

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrderLine(TypedDict):
    """One item in the order with individual pricing and retry tracking."""

    item: str
    qty: int
    price: float
    menu_status: Literal["pending", "valid", "unavailable"]
    cook_status: Literal["pending", "cooking", "done", "failed"]
    serve_status: Literal["pending", "serving", "done", "failed"]
    cook_retries: int
    serve_retries: int
    error: Optional[str]


class RestaurantState(TypedDict):
    # Conversation history. `add_messages` appends new messages instead of
    # overwriting the list, which is what lets the router "remember" earlier
    # turns (e.g. after a retry loop back from take_order).
    messages: Annotated[list[BaseMessage], add_messages]

    # The order for this turn. Empty list means "no order in this message"
    # (a question, chit-chat, etc.) — see route_after_router in edges.py.
    items: list[OrderLine]

    # Overall status flags for the cook/serve stage.
    cook_status: Literal["pending", "cooking", "done", "failed"]
    serve_status: Literal["pending", "serving", "done", "failed"]

    # Retry counters, capped in edges.py so failure loops terminate.
    cook_retries: int
    serve_retries: int

    # Total calculated bill for valid items.
    total_bill: Optional[float]

    # Overall order outcome flag for this turn.
    order_status: Optional[Literal["pending", "confirmed", "successful", "unsuccessful"]]

    # Set whenever a node wants to explain a terminal failure to the user.
    error: Optional[str]


def initial_state() -> RestaurantState:
    """A fresh state for a new order/session."""
    return RestaurantState(
        messages=[],
        items=[],
        cook_status="pending",
        serve_status="pending",
        cook_retries=0,
        serve_retries=0,
        total_bill=None,
        order_status=None,
        error=None,
    )