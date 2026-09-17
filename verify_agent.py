import os
import unittest
import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.types import Command
from graph import build_graph
import tools
import db

class TestRestaurantAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = build_graph()
        db.init_db(reset=True)

    def test_01_menu_query(self):
        print("\n--- Test 1: Menu Query ---")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = self.app.invoke({"messages": [HumanMessage(content="What's on the menu?")]}, config=config)
        reply = result["messages"][-1].content
        print(f"Reply:\n{reply[:150]}...\n")
        self.assertGreaterEqual(len(result["messages"]), 2)
        self.assertTrue(any(word in reply.lower() for word in ["pizza", "burger", "starter", "main", "menu", "drinks"]))

    def test_02_recommendation(self):
        print("\n--- Test 2: Budget Recommendation ---")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = self.app.invoke({"messages": [HumanMessage(content="I have 12 dollars, what should I get?")]}, config=config)
        reply = result["messages"][-1].content
        print(f"Reply:\n{reply[:150]}...\n")
        self.assertTrue(any(w in reply.lower() for w in ["$", "dollar", "recommend", "fit", "try", "pizza", "burger"]))

    def test_03_order_hitl_confirm_and_receipt(self):
        print("\n--- Test 3: HITL Confirmation and Receipt Calculation ---")
        orig_cook_rate = tools.COOK_FAILURE_RATE
        orig_serve_rate = tools.SERVE_FAILURE_RATE
        tools.COOK_FAILURE_RATE = 0.0
        tools.SERVE_FAILURE_RATE = 0.0
        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            # User places order: 2 burgers ($8.50 each) + 1 lemonade ($3.00) = $20.00
            self.app.invoke({"messages": [HumanMessage(content="I'd like 2 burgers and 1 lemonade")]}, config=config)
            state = self.app.get_state(config)

            # Must be interrupted for confirmation
            self.assertTrue(bool(state.tasks and any(t.interrupts for t in state.tasks)), "Graph should pause at confirmation interrupt")
            interrupt_val = state.tasks[0].interrupts[0].value
            print(f"Confirmation Prompt:\n{interrupt_val}\n")
            self.assertIn("2x burger @ $8.50 = $17.00", interrupt_val)
            self.assertIn("1x lemonade @ $3.00 = $3.00", interrupt_val)
            self.assertIn("$20.00", interrupt_val)

            # User confirms with 'yes'
            result = self.app.invoke(Command(resume="yes"), config=config)
            reply = result["messages"][-1].content
            print(f"Final Receipt Response:\n{reply}\n")
            self.assertEqual(result["order_status"], "successful")
            self.assertIn("Order Successful! Here is your receipt:", reply)
            self.assertIn("Total Bill: $20.00", reply)
            self.assertIn("2x burger @ $8.50 = $17.00", reply)
            self.assertIn("1x lemonade @ $3.00 = $3.00", reply)
        finally:
            tools.COOK_FAILURE_RATE = orig_cook_rate
            tools.SERVE_FAILURE_RATE = orig_serve_rate

    def test_04_order_hitl_cancellation(self):
        print("\n--- Test 4: HITL Order Cancellation ---")
        orig_cook_rate = tools.COOK_FAILURE_RATE
        orig_serve_rate = tools.SERVE_FAILURE_RATE
        tools.COOK_FAILURE_RATE = 0.0
        tools.SERVE_FAILURE_RATE = 0.0
        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            self.app.invoke({"messages": [HumanMessage(content="I'd like 1 burger")]}, config=config)
            state = self.app.get_state(config)
            self.assertTrue(bool(state.tasks and any(t.interrupts for t in state.tasks)))

            # User says 'no'
            result = self.app.invoke(Command(resume="no"), config=config)
            reply = result["messages"][-1].content
            print(f"Cancellation Response: {reply}")
            self.assertEqual(result["order_status"], "unsuccessful")
            self.assertIn("cancelled", reply.lower())
        finally:
            tools.COOK_FAILURE_RATE = orig_cook_rate
            tools.SERVE_FAILURE_RATE = orig_serve_rate

    def test_05_per_item_retry_tracking(self):
        print("\n--- Test 5: Per-Item Retry Tracking ---")
        orig_cook_func = tools.cook.func
        cook_attempts = {}

        def mock_cook(item: str, qty: int):
            item_clean = item.lower().strip()
            cook_attempts[item_clean] = cook_attempts.get(item_clean, 0) + 1
            # Make burger fail on attempt 1, succeed on attempt 2
            if item_clean == "burger" and cook_attempts[item_clean] == 1:
                return {"status": "failed", "reason": "grill flare-up on burger"}
            return {"status": "done", "reason": f"{qty}x {item} cooked"}

        orig_cook_rate = tools.COOK_FAILURE_RATE
        orig_serve_rate = tools.SERVE_FAILURE_RATE
        tools.COOK_FAILURE_RATE = 0.0
        tools.SERVE_FAILURE_RATE = 0.0
        tools.cook.func = mock_cook
        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            # Order 1 burger and 1 garlic bread
            self.app.invoke({"messages": [HumanMessage(content="I would like 1 burger and 1 garlic bread")]}, config=config)
            # Confirm order
            result = self.app.invoke(Command(resume="yes"), config=config)
            print("Cook attempts per item:", cook_attempts)
            # Garlic bread should be cooked only ONCE (attempt 1)
            self.assertEqual(cook_attempts.get("garlic bread"), 1, "Garlic bread should only be cooked once!")
            # Burger should be cooked TWICE (failed attempt 1, succeeded attempt 2)
            self.assertEqual(cook_attempts.get("burger"), 2, "Burger should be retried once and cooked twice!")
            self.assertEqual(result["order_status"], "successful")
            reply = result["messages"][-1].content
            print(f"Result Receipt:\n{reply}\n")
            self.assertIn("Order Successful!", reply)
        finally:
            tools.cook.func = orig_cook_func
            tools.COOK_FAILURE_RATE = orig_cook_rate
            tools.SERVE_FAILURE_RATE = orig_serve_rate

    def test_06_order_unsuccessful_out_of_stock(self):
        print("\n--- Test 6: Out of Stock Failure ---")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = self.app.invoke({"messages": [HumanMessage(content="Can I get 50 burgers?")]}, config=config)
        reply = result["messages"][-1].content
        print(f"Reply: {reply}")
        self.assertEqual(result["order_status"], "unsuccessful")
        self.assertTrue(reply.startswith("Order Unsuccessful:"))
        self.assertIn("left", reply)

    def test_07_nuance_specific_item_availability(self):
        print("\n--- Test 7: Nuanced Check (if chicken pizza is available) ---")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = self.app.invoke({"messages": [HumanMessage(content="if chicken pizza is available")]}, config=config)
        reply = result["messages"][-1].content
        print(f"Reply: {reply}")
        self.assertNotIn("Starters:\n  - garlic bread", reply)
        self.assertTrue("chicken pizza" in reply.lower() or "don't have" in reply.lower() or "not on" in reply.lower())

    def test_08_nuance_no_budget_best_dish(self):
        print("\n--- Test 8: Nuanced Best Dish (no issue of budget) ---")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = self.app.invoke({"messages": [HumanMessage(content="i have no issue of budget. just want to have the best dish here")]}, config=config)
        reply = result["messages"][-1].content
        print(f"Reply: {reply}")
        self.assertNotIn("what's your budget", reply.lower())
        self.assertNotIn("what is your budget", reply.lower())
        self.assertTrue(any(dish in reply.lower() for dish in ["chicken", "pizza", "burger", "salad", "recommend", "best"]))

if __name__ == "__main__":
    unittest.main()
