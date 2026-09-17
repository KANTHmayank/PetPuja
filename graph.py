from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from edges import (
    route_after_confirm,
    route_after_cook,
    route_after_order,
    route_after_router,
    route_after_serve,
)
from nodes import (
    confirm_order_node,
    cook_node,
    respond_node,
    router_node,
    serve_node,
    take_order_node,
)
from state import RestaurantState


def build_graph(checkpointer=None):
    graph = StateGraph(RestaurantState)

    graph.add_node("router", router_node)
    graph.add_node("take_order", take_order_node)
    graph.add_node("confirm_order", confirm_order_node)
    graph.add_node("cook", cook_node)
    graph.add_node("serve", serve_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router", route_after_router, {"take_order": "take_order", "end": END}
    )
    graph.add_conditional_edges(
        "take_order", route_after_order, {"confirm_order": "confirm_order", "respond": "respond"}
    )
    graph.add_conditional_edges(
        "confirm_order",
        route_after_confirm,
        {"cook": "cook", "router": "router", "end": END},
    )
    graph.add_conditional_edges(
        "cook", route_after_cook, {"serve": "serve", "cook": "cook", "respond": "respond"}
    )
    graph.add_conditional_edges(
        "serve", route_after_serve, {"respond": "respond", "serve": "serve"}
    )
    graph.add_edge("respond", END)

    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)



if __name__ == "__main__":
    # Sanity check: confirm every node and edge landed where expected.
    app = build_graph()
    g = app.get_graph()
    print("Nodes:", sorted(g.nodes.keys()))
    print("\nEdges:")
    for e in g.edges:
        label = f" [{e.conditional and 'cond'}]" if e.conditional else ""
        print(f"  {e.source} -> {e.target}{label}")