import os
import sys
import time
import uuid
from dotenv import load_dotenv
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

import db
import tools
tools.INITIAL_COOK_FAILURE_RATE = 0.0
tools.RETRY_COOK_FAILURE_RATE = 0.0
tools.INITIAL_SERVE_FAILURE_RATE = 0.0
tools.RETRY_SERVE_FAILURE_RATE = 0.0
from nodes import OrderItem, router_node, take_order_node
from state import initial_state
from graph import build_graph
agent_app = build_graph()
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

def test_unit_and_pricing():
    print("=== TEST 1: Unit and Pricing in db & tools ===")
    samosa_piece = db.get_piece_info("samosa")
    print("Samosa piece info:", samosa_piece)
    assert samosa_piece["has_pieces"] is True
    assert samosa_piece["price_per_piece"] == 2.00

    gj_piece = db.get_piece_info("gulab jamun")
    print("Gulab Jamun piece info:", gj_piece)
    assert gj_piece["has_pieces"] is True
    assert gj_piece["price_per_piece"] == 2.25

    rasmalai_piece = db.get_piece_info("rasmalai")
    print("Rasmalai piece info:", rasmalai_piece)
    assert rasmalai_piece["has_pieces"] is True
    assert rasmalai_piece["price_per_piece"] == 2.75

    # Test take_order with piece items
    res_gj = tools.take_order.invoke({"item": "gulab jamun (pcs)", "qty": 5})
    print("take_order 5 pcs gulab jamun:", res_gj)
    assert res_gj["status"] == "valid"
    assert res_gj["price"] == 2.25
    assert res_gj["qty"] == 5

    res_sam = tools.take_order.invoke({"item": "3 pcs samosa", "qty": 1})
    print("take_order 3 pcs samosa:", res_sam)
    assert res_sam["status"] == "valid"
    assert res_sam["price"] == 2.00
    assert res_sam["qty"] == 3

    print(">>> Test 1 passed!\n")

def test_full_graph_conversation():
    print("=== TEST 2: Multi-turn Conversation & Interrupt Test ===")
    thread_id = f"test-piece-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1: User asks how much 5 pieces of gulab jamun would cost
    print("--- Turn 1: Inquiry ---")
    msg1 = "if one serving of gulab jamun has 2 pieces, how much would 5 pieces cost"
    res1 = agent_app.invoke(
        {
            "messages": [HumanMessage(content=msg1)],
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
    last_reply = res1["messages"][-1].content
    print("Agent Reply Turn 1:\n", last_reply)
    time.sleep(2.0)

    # Turn 2: User says "okay so i would like to order 5 pieces"
    print("\n--- Turn 2: Order 5 pieces ---")
    msg2 = "okay so i would like to order 5 pieces"
    res2 = agent_app.invoke(
        {
            "messages": [HumanMessage(content=msg2)],
        },
        config=config,
    )
    print("res2 output:\n", res2)
    state2 = agent_app.get_state(config)
    assert state2.tasks and any(t.interrupts for t in state2.tasks), "Graph should be interrupted for confirmation"
    interrupt_text = state2.tasks[0].interrupts[0].value
    print("Interrupt Prompt:\n", interrupt_text)
    
    # Check items in state
    items = state2.values.get("items", [])
    total_bill = state2.values.get("total_bill")
    print("Items:", items)
    print("Total Bill:", total_bill)

    assert len(items) == 1
    assert "pcs" in items[0]["item"].lower()
    assert items[0]["qty"] == 5
    assert items[0]["price"] == 2.25
    assert total_bill == 11.25, f"Expected $11.25, got {total_bill}"
    print(">>> 5 pieces of Gulab Jamun accurately priced at $11.25!\n")

    # Turn 3: Confirm the order
    print("--- Turn 3: Confirm order ---")
    res3 = agent_app.invoke(Command(resume="yes"), config=config)
    final_reply = res3["messages"][-1].content
    print("Final Agent Receipt:\n", final_reply)
    assert "11.25" in final_reply, f"Receipt should show $11.25 total bill, got:\n{final_reply}"
    print(">>> Test 2 passed!\n")

def test_samosa_and_rasmalai():
    print("=== TEST 3: Samosa 3 pcs and Rasmalai 5 pcs ===")
    time.sleep(2.0)
    
    # Test Samosa 3 pcs
    thread_id = f"test-sam-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    res = agent_app.invoke(
        {
            "messages": [HumanMessage(content="I would like to order 3 pieces of samosa")],
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
    state = agent_app.get_state(config)
    items = state.values.get("items", [])
    total_bill = state.values.get("total_bill")
    print("Samosa items:", items)
    print("Samosa total bill:", total_bill)
    assert len(items) == 1
    assert items[0]["qty"] == 3
    assert items[0]["price"] == 2.00
    assert total_bill == 6.00, f"Expected $6.00 for 3 pcs samosa, got {total_bill}"

    # Test Rasmalai 5 pcs
    time.sleep(2.0)
    thread_id2 = f"test-ras-{uuid.uuid4()}"
    config2 = {"configurable": {"thread_id": thread_id2}}
    res2 = agent_app.invoke(
        {
            "messages": [HumanMessage(content="Can I get 5 pieces of rasmalai")],
            "items": [],
            "cook_status": "pending",
            "serve_status": "pending",
            "cook_retries": 0,
            "serve_retries": 0,
            "total_bill": None,
            "order_status": None,
            "error": None,
        },
        config=config2,
    )
    state2 = agent_app.get_state(config2)
    items2 = state2.values.get("items", [])
    total_bill2 = state2.values.get("total_bill")
    print("Rasmalai items:", items2)
    print("Rasmalai total bill:", total_bill2)
    assert len(items2) == 1
    assert items2[0]["qty"] == 5
    assert items2[0]["price"] == 2.75
    assert total_bill2 == 13.75, f"Expected $13.75 for 5 pcs rasmalai, got {total_bill2}"

    print(">>> Test 3 passed!\n")

def test_combination_and_portion_regression():
    print("=== TEST 4: Combination and Portion Regression ===")
    time.sleep(2.0)
    thread_id = f"test-combo-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    res = agent_app.invoke(
        {
            "messages": [HumanMessage(content="I want 1 chicken biryani and 3 pcs samosa")],
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
    state = agent_app.get_state(config)
    items = state.values.get("items", [])
    total_bill = state.values.get("total_bill")
    print("Combo items:", items)
    print("Combo total bill:", total_bill)
    assert len(items) == 2
    # 1 biryani @ 13.00, 3 samosa pcs @ 2.00 = 6.00 -> total = 19.00
    assert total_bill == 19.00, f"Expected $19.00, got {total_bill}"

    # Test explicit plates/portions: "2 plates of samosa and 1 beer"
    time.sleep(2.0)
    thread_id2 = f"test-portion-{uuid.uuid4()}"
    config2 = {"configurable": {"thread_id": thread_id2}}
    res2 = agent_app.invoke(
        {
            "messages": [HumanMessage(content="I'll have 2 plates of samosa and 1 beer")],
            "items": [],
            "cook_status": "pending",
            "serve_status": "pending",
            "cook_retries": 0,
            "serve_retries": 0,
            "total_bill": None,
            "order_status": None,
            "error": None,
        },
        config=config2,
    )
    state2 = agent_app.get_state(config2)
    items2 = state2.values.get("items", [])
    total_bill2 = state2.values.get("total_bill")
    print("Plates items:", items2)
    print("Plates total bill:", total_bill2)
    assert len(items2) == 2
    # 2 plates of samosa @ 4.00 = 8.00, 1 beer @ 6.50 = 6.50 -> total = 14.50
    assert total_bill2 == 14.50, f"Expected $14.50 for 2 plates of samosa + beer, got {total_bill2}"

    # Also test discrete count "2 samosas" (which is 2 pieces @ 2.00 = $4.00 + $6.50 = $10.50)
    thread_id3 = f"test-discrete-{uuid.uuid4()}"
    config3 = {"configurable": {"thread_id": thread_id3}}
    res3 = agent_app.invoke(
        {
            "messages": [HumanMessage(content="I'll have 2 samosas and 1 beer")],
            "items": [],
            "cook_status": "pending",
            "serve_status": "pending",
            "cook_retries": 0,
            "serve_retries": 0,
            "total_bill": None,
            "order_status": None,
            "error": None,
        },
        config=config3,
    )
    state3 = agent_app.get_state(config3)
    total_bill3 = state3.values.get("total_bill")
    print("Discrete '2 samosas' total bill:", total_bill3)
    assert total_bill3 == 14.50, f"Expected $14.50 for 2 samosas + beer, got {total_bill3}"
    print(">>> Test 4 passed!\n")

if __name__ == "__main__":
    db.init_db()
    test_unit_and_pricing()
    test_full_graph_conversation()
    test_samosa_and_rasmalai()
    test_combination_and_portion_regression()
    print("ALL TESTS PASSED SUCCESSFULLY! 🎉")
