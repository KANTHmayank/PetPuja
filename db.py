"""
SQLite-backed menu storage.

Replaces the old in-memory MENU dict from tools.py with a real table so the
menu can hold categories, prices, and descriptions, and grow past a handful
of items. The DB file lives next to this script so it works regardless of
the current working directory the app is launched from.
"""

import difflib
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# On Vercel / serverless environment, the deployment root directory is read-only.
# We detect Vercel or read-only filesystem and use the system temp directory.
if os.environ.get("VERCEL") or not os.access(BASE_DIR, os.W_OK):
    DB_PATH = os.path.join(tempfile.gettempdir(), "restaurant.db")
    bundled_db = os.path.join(BASE_DIR, "restaurant.db")
    if not os.path.exists(DB_PATH) and os.path.exists(bundled_db):
        try:
            shutil.copy2(bundled_db, DB_PATH)
        except Exception:
            pass
else:
    DB_PATH = os.path.join(BASE_DIR, "restaurant.db")

# (name, category, price, stock, description, image_url, is_veg, spice_level, alcohol_pct, rating, rating_count)
SEED_MENU = [
    # Starters (चटपटे स्टार्टर्स)
    ("samosa", "starter", 60.0, 15, "crispy golden pastry filled with spiced potatoes, peas, served with mint & tamarind chutney (2 pcs)", "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=400&q=80", 1, "Medium", None, 4.8, 340),
    ("chicken tikka", "starter", 280.0, 14, "smoky tandoor-roasted chicken chunks marinated in hung curd, ginger-garlic, and kashmiri spices", "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=400&q=80", 0, "Medium", None, 4.9, 420),
    ("paneer tikka", "starter", 240.0, 12, "char-grilled cottage cheese cubes with bell peppers, onions, and tangy mint dip", "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?auto=format&fit=crop&w=400&q=80", 1, "Medium", None, 4.7, 280),
    ("veg chowmein", "starter", 160.0, 16, "desi street-style wok-tossed noodles with shredded cabbage, carrots, bell peppers, and chili soy", "https://images.unsplash.com/photo-1585032226651-759b368d7246?auto=format&fit=crop&w=400&q=80", 1, "Medium", None, 4.5, 190),
    ("chicken chowmein", "starter", 210.0, 12, "wok-fried noodles with tender chicken strips, egg, spring onions, and garlic chili glaze", "https://images.unsplash.com/photo-1612927601601-6638404737ce?auto=format&fit=crop&w=400&q=80", 0, "Hot", None, 4.6, 210),
    ("dahi ke kabab", "starter", 220.0, 8, "melt-in-mouth spiced hung curd patties with a golden crispy crust, served with green chutney", "https://images.unsplash.com/photo-1541518763669-27fef04b14ea?auto=format&fit=crop&w=400&q=80", 1, "Mild", None, 4.7, 150),
    ("chicken wings", "starter", 260.0, 15, "crispy tandoori wings tossed in spicy-sweet house masala glaze", "https://images.unsplash.com/photo-1567620832903-9fc6debc209f?auto=format&fit=crop&w=400&q=80", 0, "Hot", None, 4.8, 260),
    ("spring rolls", "starter", 150.0, 10, "crispy vegetable spring rolls with sweet chili garlic dip", "https://images.unsplash.com/photo-1544025162-d76694265947?auto=format&fit=crop&w=400&q=80", 1, "Mild", None, 4.5, 140),
    ("garlic bread", "starter", 140.0, 12, "toasted herb baguette with warm garlic butter", "https://images.unsplash.com/photo-1619535860434-ba1d8fa12536?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.4, 120),

    # Mains (मुख्य भोजन)
    ("butter chicken", "main", 380.0, 10, "tandoori chicken pieces simmered in rich, velvety tomato, cream, and cashew butter gravy", "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?auto=format&fit=crop&w=400&q=80", 0, "Mild", None, 4.9, 680),
    ("paneer butter masala", "main", 320.0, 12, "fresh cottage cheese cubes bathed in silky aromatic tomato butter gravy", "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?auto=format&fit=crop&w=400&q=80", 1, "Mild", None, 4.8, 510),
    ("dal makhani", "main", 260.0, 14, "slow-cooked black lentils simmered overnight with butter, cream, and subtle fenugreek", "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=400&q=80", 1, "Mild", None, 4.9, 590),
    ("chicken biryani", "main", 340.0, 10, "royal dum-cooked basmati rice layered with marinated chicken, saffron, browned onions, served with mint raita", "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?auto=format&fit=crop&w=400&q=80", 0, "Medium", None, 4.9, 620),
    ("veg dum biryani", "main", 260.0, 8, "fragrant saffron basmati rice layered with garden vegetables, mint, and fried onions with raita", "https://images.unsplash.com/photo-1642821373181-696a54913e9a?auto=format&fit=crop&w=400&q=80", 1, "Medium", None, 4.7, 310),
    ("rogan josh", "main", 440.0, 0, "slow-braised kashmiri tender lamb shank with aromatic spices [chef specialty, out of stock today]", "https://images.unsplash.com/photo-1545247181-516773cae754?auto=format&fit=crop&w=400&q=80", 0, "Hot", None, 4.8, 240),
    ("garlic naan", "main", 60.0, 30, "freshly baked clay oven flatbread brushed with garlic butter and fresh coriander", "https://images.unsplash.com/photo-1626074353765-517a681e40be?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.8, 450),
    ("butter naan", "main", 50.0, 35, "soft, pillowy tandoori flatbread topped with melting desi butter", "https://images.unsplash.com/photo-1601050690117-94f5f6fa8bd7?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.7, 380),
    ("burger", "main", 180.0, 8, "juicy grilled patty with cheddar, crisp lettuce, and spicy tandoori mayo", "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?auto=format&fit=crop&w=400&q=80", 0, "Medium", None, 4.5, 170),
    ("margherita pizza", "main", 280.0, 6, "stone-baked pizza with rich tomato sauce, fresh mozzarella, and basil", "https://images.unsplash.com/photo-1604382355076-af4b0eb60143?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.6, 220),
    ("pepperoni pizza", "main", 360.0, 6, "classic crisp crust pizza with spicy pepperoni and bubbling mozzarella", "https://images.unsplash.com/photo-1628840042765-356cda07504e?auto=format&fit=crop&w=400&q=80", 0, "Medium", None, 4.7, 195),

    # Desserts (मीठा और मिष्ठान)
    ("gulab jamun", "dessert", 80.0, 15, "warm golden milk dumplings soaked in fragrant rose and cardamom syrup (2 pcs)", "https://images.unsplash.com/photo-1667191167440-e22295556272?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.9, 390),
    ("rasmalai", "dessert", 120.0, 12, "soft spongy cottage cheese patties steeped in thickened saffron-pistachio milk (2 pcs)", "https://images.unsplash.com/photo-1605197150742-df2168926978?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.9, 410),
    ("chocolate brownie", "dessert", 160.0, 10, "fudgy warm walnut brownie served with a scoop of vanilla ice cream", "https://images.unsplash.com/photo-1606313564200-e75d5e30476c?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.7, 230),
    ("kheer", "dessert", 90.0, 8, "traditional rice pudding slow-simmered with milk, cardamom, almonds, and saffron", "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.8, 180),
    ("ice cream sundae", "dessert", 140.0, 14, "three scoops of rich ice cream with chocolate sauce and roasted nuts", "https://images.unsplash.com/photo-1563805042-7684c019e1cb?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.6, 160),

    # Bar & Spirits (बार और कॉकटेल)
    ("liit", "bar", 380.0, 15, "long island iced tea: gin, vodka, white rum, tequila, triple sec, lemon juice, splash of cola", "https://images.unsplash.com/photo-1551024709-8f23befc6f87?auto=format&fit=crop&w=400&q=80", 1, None, 22.0, 4.9, 350),
    ("whiskey", "bar", 320.0, 20, "premium scotch / single malt (60ml) served neat, with ice, or splash of water", "https://images.unsplash.com/photo-1527061011665-3652c757a4d4?auto=format&fit=crop&w=400&q=80", 1, None, 42.8, 4.8, 290),
    ("beer", "bar", 220.0, 30, "chilled pint of premium lager (Kingfisher Ultra / Bira 91 White)", "https://images.unsplash.com/photo-1608270546103-979a4a754167?auto=format&fit=crop&w=400&q=80", 1, None, 5.0, 4.7, 440),
    ("old fashioned", "bar", 360.0, 12, "bourbon whiskey gently stirred with angostura bitters, orange peel, and cane sugar", "https://images.unsplash.com/photo-1514362545857-3bc16c4c7d1b?auto=format&fit=crop&w=400&q=80", 1, None, 32.0, 4.8, 180),
    ("mojito", "bar", 240.0, 18, "white rum crushed with fresh mint, lime juice, brown sugar, and sparkling soda", "https://images.unsplash.com/photo-1513558161293-cdaf765ed2fd?auto=format&fit=crop&w=400&q=80", 1, None, 10.0, 4.7, 270),
    ("spicy guava cocktail", "bar", 260.0, 14, "vodka spiked with pink guava juice, tabasco dash, and roasted black salt rim", "https://images.unsplash.com/photo-1536935338788-846bb9981813?auto=format&fit=crop&w=400&q=80", 1, None, 12.0, 4.8, 220),

    # Drinks & Refreshers (पेय पदार्थ)
    ("mango lassi", "drink", 110.0, 20, "thick creamy chilled yogurt smoothie blended with alphonso mango and green cardamom", "https://images.unsplash.com/photo-1626074353765-517a681e40be?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.9, 510),
    ("masala chai", "drink", 50.0, 25, "freshly brewed spiced assam tea with crushed ginger, cardamom, and milk", "https://images.unsplash.com/photo-1576092768241-dec231879fc3?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.9, 640),
    ("nimbu soda", "drink", 60.0, 20, "fresh lime sparkling soda with roasted cumin, mint, and black salt (sweet or salted)", "https://images.unsplash.com/photo-1513558161293-cdaf765ed2fd?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.6, 190),
    ("iced tea", "drink", 90.0, 16, "chilled black tea with fresh lemon slices and mint sprig", "https://images.unsplash.com/photo-1556679343-c7306c1976bc?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.5, 150),
    ("coffee", "drink", 80.0, 20, "south indian style frothy filter coffee or classic espresso", "https://images.unsplash.com/photo-1509042239860-f550ce710b93?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.6, 210),
    ("coke", "drink", 40.0, 30, "classic 330ml chilled can", "https://images.unsplash.com/photo-1622483767028-3f66f32aef97?auto=format&fit=crop&w=400&q=80", 1, None, None, 4.5, 280),
]

CATEGORY_ORDER = ["starter", "main", "dessert", "bar", "drink"]



@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
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

        # Check if table exists and has new schema columns
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='menu_items'"
        ).fetchone()
        if table_exists:
            cols = [c[1] for c in conn.execute("PRAGMA table_info(menu_items)").fetchall()]
            if "image_url" not in cols or "rating" not in cols:
                conn.execute("DROP TABLE IF EXISTS menu_items")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS menu_items (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL,
                description TEXT,
                image_url TEXT,
                is_veg INTEGER DEFAULT 1,
                spice_level TEXT,
                alcohol_pct REAL,
                rating REAL DEFAULT 4.5,
                rating_count INTEGER DEFAULT 100
            )
            """
        )
        (count,) = conn.execute("SELECT COUNT(*) FROM menu_items").fetchone()
        if count == 0:
            conn.executemany(
                "INSERT INTO menu_items (name, category, price, stock, description, image_url, is_veg, spice_level, alcohol_pct, rating, rating_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                SEED_MENU,
            )


def normalize_name(name: str) -> str:
    """Strip leading numbers, unit phrases like 'pcs', 'plates', 'glasses', etc."""
    cleaned = name.lower().strip()
    # Strip unit words with or without numbers (e.g., '3 pcs samosa' -> 'samosa', 'plate of veg chowmein' -> 'veg chowmein')
    cleaned = re.sub(
        r"^\s*(\d+\s*)?(pcs|pc|pieces|piece|plates|plate|portions|portion|glasses|glass|bottles|bottle|cans|can|bowls|bowl|servings|serving|pints|pint)\s*(of\s+)?(the\s+)?",
        "",
        cleaned,
    ).strip()
    # Strip any remaining leading digits (e.g., '3 samosas' -> 'samosas')
    cleaned = re.sub(r"^\s*\d+\s*(of\s+)?(the\s+)?", "", cleaned).strip()
    # Strip trailing unit words
    cleaned = re.sub(
        r"\s+(pcs|pc|pieces|piece|plates|plate|portions|portion|glasses|glass|pints|pint)$",
        "",
        cleaned,
    ).strip()
    return cleaned


def get_item(name: str) -> sqlite3.Row | None:
    """Find a menu item by exact name, normalized name, plural/singular variation,
    substring, or fuzzy match."""
    raw = name.lower().strip()
    with _connect() as conn:
        # 1. Exact match
        row = conn.execute("SELECT * FROM menu_items WHERE name = ?", (raw,)).fetchone()
        if row is not None:
            return row

        # 2. Normalized match (without prefixes like '3 pcs', 'plate of', etc.)
        norm = normalize_name(raw)
        if norm:
            row = conn.execute("SELECT * FROM menu_items WHERE name = ?", (norm,)).fetchone()
            if row is not None:
                return row

        # 3. Singular / plural matching
        target = norm or raw
        if target.endswith("s") and len(target) > 3:
            singular = target[:-1]
            row = conn.execute("SELECT * FROM menu_items WHERE name = ?", (singular,)).fetchone()
            if row is not None:
                return row
            if target.endswith("es") and len(target) > 4:
                singular_es = target[:-2]
                row = conn.execute("SELECT * FROM menu_items WHERE name = ?", (singular_es,)).fetchone()
                if row is not None:
                    return row
        else:
            row = conn.execute("SELECT * FROM menu_items WHERE name = ?", (target + "s",)).fetchone()
            if row is not None:
                return row

        # 4. Substring matching against all menu items (e.g. 'samosa' inside '3 pcs samosa')
        all_rows = conn.execute("SELECT * FROM menu_items").fetchall()
        for r in all_rows:
            item_name = r["name"]
            if target == item_name or target in item_name or item_name in target:
                return r

        # 5. Close fuzzy matching
        name_map = {r["name"]: r for r in all_rows}
        close = difflib.get_close_matches(target, list(name_map.keys()), n=1, cutoff=0.7)
        if close:
            return name_map[close[0]]

        return None


def get_all_names() -> list[str]:
    with _connect() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM menu_items")]


def get_piece_info(item_name: str) -> dict:
    """Check if an item has piece-level pricing (e.g., 2 pcs per portion).
    Returns a dict with pieces_per_portion, price_per_piece, and portion_price.
    """
    clean = re.sub(r"\(pcs\)|\(pieces\)|\(pc\)", "", item_name.lower()).strip()
    row = get_item(clean)
    if not row:
        return {"has_pieces": False, "pieces_per_portion": 1, "price_per_piece": 0.0, "portion_price": 0.0, "name": clean}

    desc = row["description"].lower()
    match = re.search(r"\((\d+)\s*(?:pcs|pc|pieces|piece)\)", desc)
    if match:
        pcs = int(match.group(1))
        if pcs > 0:
            price_per_pc = round(row["price"] / pcs, 2)
            return {
                "has_pieces": True,
                "pieces_per_portion": pcs,
                "price_per_piece": price_per_pc,
                "portion_price": row["price"],
                "name": row["name"],
            }
    return {
        "has_pieces": False,
        "pieces_per_portion": 1,
        "price_per_piece": row["price"],
        "portion_price": row["price"],
        "name": row["name"],
    }


def decrement_stock(name: str, qty: int) -> None:
    clean = re.sub(r"\(pcs\)|\(pieces\)|\(pc\)", "", name.lower()).strip()
    with _connect() as conn:
        conn.execute(
            "UPDATE menu_items SET stock = MAX(0, stock - ?) WHERE name = ?",
            (qty, clean),
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
            lines.append(f"  - {r['name']} (₹{r['price']:.0f}) — {r['description']} [{stock_note}]")
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