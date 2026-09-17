"""
SQLite-backed menu storage.

Replaces the old in-memory MENU dict from tools.py with a real table so the
menu can hold categories, prices, and descriptions, and grow past a handful
of items. The DB file lives next to this script so it works regardless of
the current working directory the app is launched from.
"""

import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restaurant.db")

# (name, category, price, stock, description)
# (name, category, price, stock, description)
SEED_MENU = [
    # Starters (चटपटे स्टार्टर्स)
    ("samosa", "starter", 4.00, 15, "crispy golden pastry filled with spiced potatoes, peas, served with mint & tamarind chutney (2 pcs)"),
    ("chicken tikka", "starter", 8.50, 14, "smoky tandoor-roasted chicken chunks marinated in hung curd, ginger-garlic, and kashmiri spices"),
    ("paneer tikka", "starter", 7.50, 12, "char-grilled cottage cheese cubes with bell peppers, onions, and tangy mint dip"),
    ("veg chowmein", "starter", 6.50, 16, "desi street-style wok-tossed noodles with shredded cabbage, carrots, bell peppers, and chili soy"),
    ("chicken chowmein", "starter", 8.00, 12, "wok-fried noodles with tender chicken strips, egg, spring onions, and garlic chili glaze"),
    ("dahi ke kabab", "starter", 7.00, 8, "melt-in-mouth spiced hung curd patties with a golden crispy crust, served with green chutney"),
    ("chicken wings", "starter", 7.50, 15, "crispy tandoori wings tossed in spicy-sweet house masala glaze"),
    ("spring rolls", "starter", 5.50, 10, "crispy vegetable spring rolls with sweet chili garlic dip"),
    ("garlic bread", "starter", 4.50, 12, "toasted herb baguette with warm garlic butter"),

    # Mains (मुख्य भोजन)
    ("butter chicken", "main", 12.50, 10, "tandoori chicken pieces simmered in rich, velvety tomato, cream, and cashew butter gravy"),
    ("paneer butter masala", "main", 11.00, 12, "fresh cottage cheese cubes bathed in silky aromatic tomato butter gravy"),
    ("dal makhani", "main", 9.50, 14, "slow-cooked black lentils simmered overnight with butter, cream, and subtle fenugreek"),
    ("chicken biryani", "main", 13.00, 10, "royal dum-cooked basmati rice layered with marinated chicken, saffron, browned onions, served with mint raita"),
    ("veg dum biryani", "main", 10.50, 8, "fragrant saffron basmati rice layered with garden vegetables, mint, and fried onions with raita"),
    ("rogan josh", "main", 14.50, 0, "slow-braised kashmiri tender lamb shank with aromatic spices [chef specialty, out of stock today]"),
    ("garlic naan", "main", 3.50, 30, "freshly baked clay oven flatbread brushed with garlic butter and fresh coriander"),
    ("butter naan", "main", 3.00, 35, "soft, pillowy tandoori flatbread topped with melting desi butter"),
    ("burger", "main", 8.50, 8, "juicy grilled patty with cheddar, crisp lettuce, and spicy tandoori mayo"),
    ("margherita pizza", "main", 9.50, 6, "stone-baked pizza with rich tomato sauce, fresh mozzarella, and basil"),
    ("pepperoni pizza", "main", 10.50, 6, "classic crisp crust pizza with spicy pepperoni and bubbling mozzarella"),

    # Desserts (मीठा और मिष्ठान)
    ("gulab jamun", "dessert", 4.50, 15, "warm golden milk dumplings soaked in fragrant rose and cardamom syrup (2 pcs)"),
    ("rasmalai", "dessert", 5.50, 12, "soft spongy cottage cheese patties steeped in thickened saffron-pistachio milk"),
    ("chocolate brownie", "dessert", 4.50, 10, "fudgy warm walnut brownie served with a scoop of vanilla ice cream"),
    ("kheer", "dessert", 4.00, 8, "traditional rice pudding slow-simmered with milk, cardamom, almonds, and saffron"),
    ("ice cream sundae", "dessert", 4.50, 14, "three scoops of rich ice cream with chocolate sauce and roasted nuts"),

    # Bar & Spirits (बार और कॉकटेल)
    ("liit", "bar", 12.00, 15, "long island iced tea: gin, vodka, white rum, tequila, triple sec, lemon juice, splash of cola"),
    ("whiskey", "bar", 10.00, 20, "premium scotch / single malt (60ml) served neat, with ice, or splash of water"),
    ("beer", "bar", 6.50, 30, "chilled pint of premium lager (Kingfisher Ultra / Bira 91 White)"),
    ("old fashioned", "bar", 11.00, 12, "bourbon whiskey gently stirred with angostura bitters, orange peel, and cane sugar"),
    ("mojito", "bar", 8.50, 18, "white rum crushed with fresh mint, lime juice, brown sugar, and sparkling soda"),
    ("spicy guava cocktail", "bar", 9.50, 14, "vodka spiked with pink guava juice, tabasco dash, and roasted black salt rim"),

    # Drinks & Refreshers (पेय पदार्थ)
    ("mango lassi", "drink", 4.00, 20, "thick creamy chilled yogurt smoothie blended with alphonso mango and green cardamom"),
    ("masala chai", "drink", 3.00, 25, "freshly brewed spiced assam tea with crushed ginger, cardamom, and milk"),
    ("nimbu soda", "drink", 3.00, 20, "fresh lime sparkling soda with roasted cumin, mint, and black salt (sweet or salted)"),
    ("iced tea", "drink", 3.50, 16, "chilled black tea with fresh lemon slices and mint sprig"),
    ("coffee", "drink", 3.50, 20, "south indian style frothy filter coffee or classic espresso"),
    ("coke", "drink", 2.50, 30, "classic 330ml chilled can"),
]

CATEGORY_ORDER = ["starter", "main", "dessert", "bar", "drink"]



@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(reset: bool = False) -> None:
    """Create the table and seed it if empty. Idempotent — safe to call on
    every startup. Pass reset=True to wipe and reseed from scratch."""
    with _connect() as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS menu_items")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS menu_items (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL,
                description TEXT
            )
            """
        )
        (count,) = conn.execute("SELECT COUNT(*) FROM menu_items").fetchone()
        if count == 0:
            conn.executemany(
                "INSERT INTO menu_items (name, category, price, stock, description) "
                "VALUES (?, ?, ?, ?, ?)",
                SEED_MENU,
            )


def get_item(name: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM menu_items WHERE name = ?", (name.lower().strip(),)
        ).fetchone()


def get_all_names() -> list[str]:
    with _connect() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM menu_items")]


def decrement_stock(name: str, qty: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE menu_items SET stock = MAX(0, stock - ?) WHERE name = ?",
            (qty, name.lower().strip()),
        )


def format_menu_text() -> str:
    """Grouped, human-readable menu — always queried fresh so stock/prices
    are never stale."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM menu_items ORDER BY category, name"
        ).fetchall()

    by_category: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    lines = []
    for category in CATEGORY_ORDER:
        items = by_category.get(category)
        if not items:
            continue
        lines.append(f"\n{category.capitalize()}s:")
        for r in items:
            stock_note = f"{r['stock']} left" if r["stock"] > 0 else "out of stock"
            lines.append(f"  - {r['name']} (${r['price']:.2f}) — {r['description']} [{stock_note}]")
    return "\n".join(lines).strip()


def get_available() -> list[sqlite3.Row]:
    """In-stock items only — the pool a recommendation can be built from."""
    with _connect() as conn:
        return conn.execute("SELECT * FROM menu_items WHERE stock > 0").fetchall()


def suggest_combo(budget: float, sizes: tuple[int, ...] = (3, 2, 1)) -> tuple[list[sqlite3.Row], float] | None:
    """Best in-stock combo of distinct items that fits the budget.

    Deliberately not left to the LLM: it's a small, exact combinatorics
    problem (the menu is small enough to brute-force), and price arithmetic
    is exactly the kind of thing a language model can get subtly wrong.
    Tries 3-item combos first, then 2, then 1, and within a size picks the
    one that uses the most of the budget (closest to it without going over).
    """
    from itertools import combinations

    items = get_available()
    for size in sizes:
        if len(items) < size:
            continue
        best_combo, best_total = None, -1.0
        for combo in combinations(items, size):
            total = sum(r["price"] for r in combo)
            if total <= budget and total > best_total:
                best_combo, best_total = combo, total
        if best_combo is not None:
            return list(best_combo), best_total
    return None


init_db()