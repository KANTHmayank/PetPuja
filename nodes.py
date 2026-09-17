"""
Graph nodes. Only `router_node` calls the LLM — the rest are thin wrappers
around the deterministic functions in tools.py, so they're cheap and fast.
"""

from functools import lru_cache
import re
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.types import interrupt

from state import RestaurantState
from tools import cook, serve, take_order
import db

MAX_COOK_RETRIES = 2
MAX_SERVE_RETRIES = 2

GROQ_MODEL = "openai/gpt-oss-20b"


class OrderItem(BaseModel):
    item: str = Field(description="the food or drink item requested, matched to the closest menu item name if possible")
    quantity: int = Field(default=1, description="how many units requested; default to 1 if unspecified")


class RouterOutput(BaseModel):
    """What the router asks Groq to extract or compose based on the conversation."""

    is_order: bool = Field(
        description=(
            "True ONLY if the customer is specifically placing an order to buy/eat food or drink right now "
            "(e.g. 'I want 2 burgers', 'I'd like to order pizza', 'give me a coke', 'can I get garlic bread', 'I'll have the burger'). "
            "False for questions, checking availability (e.g. 'is chicken pizza available'), asking for recommendations, "
            "discussing budget, asking what drinks/starters exist, greetings, or casual chat."
        )
    )
    items: list[OrderItem] = Field(
        default_factory=list,
        description="List of food/drink items the customer wants to order right now. Populate ONLY when is_order is True.",
    )
    response: Optional[str] = Field(
        default=None,
        description=(
            "When is_order is False, provide a friendly, helpful, natural response acting strictly as Ramoo Kaka, PetPuja's restaurant host. "
            "Use the live menu details to answer dining, food, drink, and bar inquiries. "
            "NEVER answer coding problems, LeetCode, algorithms, math homework, or non-restaurant tasks — politely decline in character and redirect to our food and drinks. "
            "When is_order is True, leave this empty/None."
        ),
    )


MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def invoke_router_llm(prompt_messages: list) -> RouterOutput:
    """Invoke structured output with automatic multi-model fallback on rate limits."""
    last_error = None
    for model_name in MODELS:
        try:
            llm = ChatGroq(model=model_name, temperature=0.2, max_retries=2).with_structured_output(
                RouterOutput, method="json_schema"
            )
            return llm.invoke(prompt_messages)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # If rate limit or token quota exceeded, try next fallback model
            if "rate_limit" in error_str or "429" in error_str or "tpd" in error_str:
                continue

            # Check if failed_generation was provided in the error body
            failed_gen = None
            if hasattr(e, "body") and isinstance(e.body, dict):
                failed_gen = e.body.get("error", {}).get("failed_generation")
            if not failed_gen and hasattr(e, "response") and hasattr(e.response, "json"):
                try:
                    failed_gen = e.response.json().get("error", {}).get("failed_generation")
                except Exception:
                    pass
            if failed_gen:
                return RouterOutput(is_order=False, items=[], response=failed_gen.strip())
            continue

    if last_error:
        error_str = str(last_error).lower()
        if "rate_limit" in error_str or "429" in error_str or "tpd" in error_str:
            return RouterOutput(
                is_order=False,
                items=[],
                response=(
                    "Namaste! 🙏 Our kitchen and bar are experiencing a high rush of orders right now (API rate limit). "
                    "Please wait a brief moment and Ramoo Kaka will be right with you!"
                ),
            )
        raise last_error
    return RouterOutput(is_order=False, items=[], response="Namaste! 🙏 How can I assist you with our menu today?")


def router_node(state: RestaurantState) -> dict:
    """Classify customer intent and handle inquiries with culinary nuance or route orders to the kitchen."""
    menu_details = db.format_menu_text()
    system_prompt = f"""You are "Ramoo Kaka", the warm, charming, and food-loving host and maitre d' at "PetPuja" (पेटपूजा) — a premier smart Indian bistro & bar.
You embody the true spirit of Indian culinary hospitality ("Atithi Devo Bhava" — the guest is honored like God). You speak fluent, warm English with a delightful Indian bistro charm (incorporating polite greetings like "Namaste! 🙏", "Swagat hai!").

You know great food and bar craft inside out:
- Traditional Starters: crispy Samosas (with mint & tamarind chutney), Chicken Tikka, Paneer Tikka, Dahi ke Kabab, Desi Chowmein (Veg / Chicken), crispy Chicken Wings, etc.
- Hearty Mains: Butter Chicken, Paneer Butter Masala, Dal Makhani slow-cooked overnight, fragrant Chicken & Veg Dum Biryani, Garlic Naan, Butter Naan, stone-baked Pizzas, and Burgers.
- Heavenly Desserts: warm Gulab Jamun, chilled saffron Rasmalai, slow-simmered Kheer, fudge Brownie with ice cream.
- Energetic Bar & Spirits: Long Island Iced Tea (LIIT), premium Whiskey/Scotch, chilled Beers (Kingfisher Ultra / Bira), Old Fashioned, classic Mojito, and zesty Spicy Guava Cocktail.
- Refreshing Beverages: Alphonso Mango Lassi, piping hot Masala Chai, spicy Nimbu Soda, Iced Tea, South Indian Filter Coffee.

Here is our live menu with prices, descriptions, and current inventory:
{menu_details}

Your responsibilities:
1. PLACING AN ORDER:
   - If the guest explicitly wants to order/buy food or drink items (e.g. "I'll have 2 samosas and a beer", "3 pcs samosa", "bring me butter chicken with garlic naan and an LIIT", "I want a whiskey and chicken tikka"):
     Set `is_order: True` and extract each item and quantity into `items`.
     IMPORTANT: Match item names to clean canonical menu names (e.g., for "3 pcs samosa" or "3 samosas", set item="samosa", quantity=3; for "2 plates of chowmein", set item="veg chowmein", quantity=2). DO NOT include numbers or unit words (like "pcs", "plates", "glasses") inside the `item` field!
     Match ambiguous items where appropriate (e.g. "chowmein" -> "veg chowmein"; "naan" -> "garlic naan"; "tikka" -> "chicken tikka"; "biryani" -> "chicken biryani"). If an item is genuinely not on our menu, keep the customer's wording so our order validation node can handle it.
     Leave `response` as None.

2. INQUIRIES, QUESTIONS, RECOMMENDATIONS, OR DINING CHAT:
   - If the guest is NOT placing an order right now, set `is_order: False`, `items: []`, and provide a helpful, natural, and delightful `response`:
     - Specific Item Availability: Check the menu. If an item isn't on the menu (e.g. "if chicken pizza is available") or is out of stock (e.g. "is rogan josh available"), explain politely with warmth and suggest delicious available alternatives (e.g. for chicken pizza, suggest our Pepperoni Pizza, or succulent Chicken Tikka / Butter Chicken!). Never dump the entire menu unless specifically asked!
     - Recommendations & Best Dishes: If asked for recommendations (e.g. "i have no issue of budget. just want to have the best dish here" or "what do you recommend?"):
       Suggest a royal PetPuja feast! For example: Start with smoky Chicken Tikka or crisp Samosas, follow with royal Butter Chicken or Paneer Butter Masala paired with crisp Garlic Naan or fragrant Chicken Dum Biryani, accompanied by an LIIT or chilled Beer, and finish with melt-in-mouth Rasmalai or Gulab Jamun. Never ask for a budget if the customer said budget is no issue!
     - Bar & Drinks Inquiries: If asked about drinks, bar options, cocktails, or spirits (e.g. "what do you have in bar?", "do you have whiskey or beer?", "what cocktails do you serve?"), highlight our signature LIIT, single malt Whiskey, chilled Beers, classic Mojito, and Spicy Guava Cocktail, plus Mango Lassi and Masala Chai for non-alcoholic guests.
     - Budget Inquiries: If the guest mentions a specific budget (e.g. "I have $15"), suggest delicious in-stock combinations or items fitting that budget (e.g., Samosa + Dal Makhani + Garlic Naan or Veg Chowmein + Mango Lassi).
     - Category Inquiries: If asked about a category (e.g. "what starters do you have?"), present only that category clearly with prices.
     - Full Menu Request: Only if the guest asks to see the entire menu ("what's on the menu?"), provide a clean overview of our categories (Starters, Mains, Desserts, Bar & Spirits, Drinks).
     - Dining banter / Greetings: Be warm, charming, and welcoming.
     - Multi-turn context: Remember earlier turns in the conversation thread and maintain natural conversational continuity.

3. STRICT GUARDRAIL FOR OFF-TOPIC & NON-RESTAURANT REQUESTS:
   - You are exclusively the dining host at PetPuja Bistro & Bar. You NEVER answer questions outside food, beverages, dining, restaurant ambiance, or reservations.
   - If a guest asks you to solve coding or software problems (e.g., LeetCode problems like "rat in a maze", "two sum", "write Python code", "debugging"), do math homework, discuss politics, write academic essays, or answer general technical/trivia questions:
     Set `is_order: False`, `items: []`.
     POLITELY AND HUMOROUSLY DECLINE in character as Ramoo Kaka! State that you are a humble restaurant maitre d' whose only algorithms are culinary recipes, and steer them warmly back to the PetPuja menu.
     Example: "Arre sahab! 🙏 I am Ramoo Kaka, your dining host at PetPuja, not a software engineer! In my kitchen, the only maze we navigate is the fragrant swirl of basmati rice in our Chicken Biryani, and we solve appetites, not backtracking algorithms! Let's leave LeetCode to the computers while I serve you some sizzling Chicken Tikka or a crisp Samosa with a chilled LIIT. What delicious dish or drink can I get you today?"
"""

    prompt_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    result: RouterOutput = invoke_router_llm(prompt_messages)

    if result.is_order and result.items:
        normalized_items = []
        for i in result.items:
            raw_name = i.item.lower().strip()
            qty = i.quantity or 1
            # If quantity is 1 but raw_name starts with a number like "3 pcs samosa", extract it
            match = re.match(r"^\s*(\d+)\s*(pcs|pc|pieces|piece|plates|plate|portions|portion|glasses|glass|bottles|bottle|cans|can|bowls|bowl|servings|serving|pints|pint)?", raw_name)
            if match and qty == 1:
                extracted_qty = int(match.group(1))
                if extracted_qty > 0:
                    qty = extracted_qty

            clean_name = db.normalize_name(raw_name) or raw_name
            db_row = db.get_item(clean_name)
            canonical_name = db_row["name"] if db_row is not None else clean_name

            normalized_items.append({
                "item": canonical_name,
                "qty": qty,
                "price": db_row["price"] if db_row is not None else 0.0,
                "menu_status": "pending",
                "cook_status": "pending",
                "serve_status": "pending",
                "cook_retries": 0,
                "serve_retries": 0,
                "error": None,
            })
        return {"items": normalized_items}

    # If it's not an order, or no items could be identified
    reply_text = result.response or "Welcome! How can I help you with our menu today?"
    return {"items": [], "messages": [AIMessage(content=reply_text)]}


def take_order_node(state: RestaurantState) -> dict:
    updated = []
    errors = []
    for line in state["items"]:
        result = take_order.invoke({"item": line["item"], "qty": line["qty"]})
        status = result["status"]
        price = result.get("price", 0.0)
        reason = result.get("reason")
        canonical_item = result.get("item", line["item"])
        qty = result.get("qty", line["qty"])
        updated.append({
            **line,
            "item": canonical_item,
            "qty": qty,
            "price": price,
            "menu_status": status,
            "cook_status": "pending",
            "serve_status": "pending",
            "cook_retries": 0,
            "serve_retries": 0,
            "error": reason if status != "valid" else None,
        })
        if status == "unavailable":
            errors.append(reason or f"'{canonical_item}' is unavailable")

    error = "; ".join(errors) if errors else None
    valid_items = [l for l in updated if l["menu_status"] == "valid"]
    total_bill = sum(l["qty"] * l["price"] for l in valid_items) if valid_items else None
    return {"items": updated, "total_bill": total_bill, "error": error}


def confirm_order_node(state: RestaurantState) -> dict:
    """Human-in-the-loop: present order summary + total bill and wait for confirmation before cooking."""
    valid_items = [l for l in state["items"] if l["menu_status"] == "valid"]
    lines = [
        f"  - {l['qty']}x {l['item']} @ ${l['price']:.2f} = ${l['qty'] * l['price']:.2f}"
        for l in valid_items
    ]
    total = sum(l["qty"] * l["price"] for l in valid_items)
    prompt = (
        "Order Summary:\n"
        + "\n".join(lines)
        + f"\nTotal: ${total:.2f}\n"
        + "Shall I place this order with the kitchen? (yes/no): "
    )
    user_resp = interrupt(prompt)
    is_confirmed = str(user_resp).strip().lower() in ["yes", "y", "sure", "ok", "confirm", "yeah", "yep"]
    if is_confirmed:
        return {
            "order_status": "confirmed",
            "total_bill": total,
        }
    else:
        return {
            "order_status": "unsuccessful",
            "error": "Order cancelled by customer",
            "messages": [AIMessage(content="Order cancelled. Let me know if you would like to order anything else!")],
        }


def cook_node(state: RestaurantState) -> dict:
    """Cook valid items with per-item retry tracking. Only uncompleted items are retried."""
    updated_items = []
    failures = []
    for line in state["items"]:
        # Only cook items that are valid and haven't completed cooking yet
        if line["menu_status"] == "valid" and line["cook_status"] != "done":
            result = cook.invoke({"item": line["item"], "qty": line["qty"]})
            if result["status"] == "done":
                updated_items.append({**line, "cook_status": "done", "error": None})
            else:
                new_retries = line["cook_retries"] + 1
                reason = result.get("reason", f"Kitchen issue cooking {line['item']}")
                updated_items.append({
                    **line,
                    "cook_status": "failed",
                    "cook_retries": new_retries,
                    "error": reason,
                })
                failures.append(f"{line['item']} ({reason})")
        else:
            # Already cooked or unavailable: leave unchanged
            updated_items.append(line)

    valid_items = [l for l in updated_items if l["menu_status"] == "valid"]
    all_cooked = all(l["cook_status"] == "done" for l in valid_items)

    if all_cooked:
        return {"items": updated_items, "cook_status": "done", "error": None}
    else:
        return {
            "items": updated_items,
            "cook_status": "failed",
            "cook_retries": state.get("cook_retries", 0) + 1,
            "error": "; ".join(failures) if failures else state.get("error"),
        }


def serve_node(state: RestaurantState) -> dict:
    """Serve cooked items with per-item retry tracking. Only unserved items are retried."""
    updated_items = []
    failures = []
    for line in state["items"]:
        # Only serve items that are cooked and haven't completed serving yet
        if line["cook_status"] == "done" and line["serve_status"] != "done":
            result = serve.invoke({"item": line["item"], "qty": line["qty"]})
            if result["status"] == "done":
                updated_items.append({**line, "serve_status": "done", "error": None})
            else:
                new_retries = line["serve_retries"] + 1
                reason = result.get("reason", f"Serving mishap with {line['item']}")
                updated_items.append({
                    **line,
                    "serve_status": "failed",
                    "serve_retries": new_retries,
                    "error": reason,
                })
                failures.append(f"{line['item']} ({reason})")
        else:
            # Already served or not cooked: leave unchanged
            updated_items.append(line)

    cooked_items = [l for l in updated_items if l["cook_status"] == "done"]
    all_served = all(l["serve_status"] == "done" for l in cooked_items)

    if all_served:
        return {"items": updated_items, "serve_status": "done", "error": None}
    else:
        return {
            "items": updated_items,
            "serve_status": "failed",
            "serve_retries": state.get("serve_retries", 0) + 1,
            "error": "; ".join(failures) if failures else state.get("error"),
        }


def respond_node(state: RestaurantState) -> dict:
    """Compose receipt or explain terminal failure reasons."""
    valid_items = [l for l in state.get("items", []) if l["menu_status"] == "valid"]
    unavailable_items = [l for l in state.get("items", []) if l["menu_status"] == "unavailable"]

    if state.get("serve_status") == "done" and valid_items:
        order_status = "successful"
        receipt_lines = [
            f"  - {l['qty']}x {l['item']} @ ${l['price']:.2f} = ${l['qty'] * l['price']:.2f}"
            for l in valid_items
        ]
        total = sum(l["qty"] * l["price"] for l in valid_items)
        divider = "-" * 42
        text = (
            "🎉 Dhanyavaad! Your order is cooked, prepared, and served fresh!\n"
            "Here is your PetPuja receipt:\n"
            f"{divider}\n"
            + "\n".join(receipt_lines)
            + f"\n{divider}\n"
            + f"Total Bill: ${total:.2f}\n\n"
            + "Swadist bhojan aur drinks ka anand lijiye! (Enjoy your delicious meal & drinks!) 🙏"
        )
    elif unavailable_items:
        order_status = "unsuccessful"
        reason = state.get("error") or "Requested item(s) are unavailable"
        text = f"Order Unsuccessful: {reason}. Would you like to order something else?"
    elif state.get("order_status") == "unsuccessful" and state.get("error") == "Order cancelled by customer":
        order_status = "unsuccessful"
        text = "Order cancelled. Let me know if you would like to order anything else!"
    elif state.get("cook_status") == "failed":
        order_status = "unsuccessful"
        failed_items = [l for l in valid_items if l["cook_status"] != "done"]
        reasons = [f"{l['item']} ({l['error']})" for l in failed_items]
        reason_str = "; ".join(reasons) if reasons else (state.get("error") or "Kitchen cooking issue")
        text = f"Order Unsuccessful: Kitchen cooking retry attempts exhausted. Reason: {reason_str}."
    elif state.get("serve_status") == "failed":
        order_status = "unsuccessful"
        failed_items = [l for l in valid_items if l["serve_status"] != "done"]
        reasons = [f"{l['item']} ({l['error']})" for l in failed_items]
        reason_str = "; ".join(reasons) if reasons else (state.get("error") or "Serving mishap")
        text = f"Order Unsuccessful: Serving retry attempts exhausted. Reason: {reason_str}."
    else:
        order_status = "unsuccessful"
        reason = state.get("error") or "An unexpected issue occurred while processing your order"
        text = f"Order Unsuccessful: {reason}."


    return {
        "order_status": order_status,
        "messages": [AIMessage(content=text)],
    }