"""Validation report for data/sqlite/backend_shaped_synthetic.db.

Read-only inspection/validation only - does not modify the database, does
not touch model code, does not train or rebuild any index.

Usage:
    python scripts/validate_backend_shaped_sqlite.py [--db PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "sqlite" / "backend_shaped_synthetic.db"


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    args = parser.parse_args()
    db_path = Path(args.db)

    con = connect_ro(db_path)
    cur = con.cursor()

    section("FILE")
    print("path:", db_path, "size_bytes:", db_path.stat().st_size)

    section("ROW COUNTS")
    tables = ["User", "Category", "Product", "Tag", "ProductTags", "Cart", "Cart_Item",
              '"Order"', "Order_Item", "Review", "UserAddress", "Voucher", "User_events"]
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"{t:15s} {cur.fetchone()[0]}")

    section("FOREIGN KEY CHECK")
    cur.execute("PRAGMA foreign_key_check")
    fk_issues = cur.fetchall()
    print("foreign_key_check violations:", len(fk_issues))
    for row in fk_issues[:20]:
        print(" ", row)

    section("PRIMARY KEY DUPLICATE CHECK")
    for t in ["User", "Category", "Product", "Tag", "ProductTags", "Cart", "Cart_Item",
              '"Order"', "Order_Item", "Review", "UserAddress", "Voucher", "User_events"]:
        idcol = "id" if t == "User_events" else "Id"
        cur.execute(f"SELECT {idcol}, COUNT(*) c FROM {t} GROUP BY {idcol} HAVING c>1")
        dups = cur.fetchall()
        if dups:
            print(f"{t}: {len(dups)} duplicate PKs!", dups[:5])
    print("(no output above other than this line = no duplicate PKs found)")

    section("USER ENTITY")
    cur.execute("SELECT COUNT(*) FROM User")
    n_users = cur.fetchone()[0]
    print("total users:", n_users)
    cur.execute("SELECT COUNT(*) FROM User WHERE PreferredCategoryId IS NULL")
    print("null PreferredCategoryId:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM User WHERE AgeGroup IS NULL")
    print("null AgeGroup:", cur.fetchone()[0])
    cur.execute("SELECT AgeGroup, COUNT(*) FROM User GROUP BY AgeGroup ORDER BY 2 DESC")
    print("AgeGroup distribution:", cur.fetchall())
    cur.execute("""SELECT c.Name, COUNT(*) FROM User u JOIN Category c ON u.PreferredCategoryId=c.Id
                   GROUP BY c.Name ORDER BY 2 DESC""")
    print("PreferredCategory distribution:", cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM User WHERE PreferredCategoryId NOT IN (SELECT Id FROM Category) AND PreferredCategoryId IS NOT NULL")
    print("dangling PreferredCategoryId:", cur.fetchone()[0])

    section("CATEGORY")
    cur.execute("SELECT COUNT(*) FROM Category")
    print("total categories:", cur.fetchone()[0])
    cur.execute("SELECT Id, ParentId, Name FROM Category ORDER BY Id")
    for r in cur.fetchall():
        print(" ", r)
    cur.execute("SELECT COUNT(*) FROM Category WHERE ParentId IS NOT NULL AND ParentId NOT IN (SELECT Id FROM Category)")
    print("orphaned ParentId:", cur.fetchone()[0])

    section("PRODUCT")
    cur.execute("SELECT COUNT(*) FROM Product")
    n_products = cur.fetchone()[0]
    print("total products:", n_products)
    cur.execute("SELECT isActive, COUNT(*) FROM Product GROUP BY isActive")
    active_dist = dict(cur.fetchall())
    print("isActive distribution:", active_dist)
    cur.execute("SELECT COUNT(*) FROM Product WHERE StockQuantity=0")
    oos = cur.fetchone()[0]
    print("out-of-stock (StockQuantity=0):", oos)
    cur.execute("SELECT COUNT(*) FROM Product WHERE isActive=1 AND StockQuantity>0")
    print("active+in-stock:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM Product WHERE isActive=1 AND StockQuantity=0")
    print("active+out-of-stock:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM Product WHERE isActive=0 AND StockQuantity>0")
    print("inactive+in-stock:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM Product WHERE isActive=0 AND StockQuantity=0")
    print("inactive+out-of-stock:", cur.fetchone()[0])

    cur.execute("SELECT Price FROM Product")
    prices = np.array([r[0] for r in cur.fetchall()])
    print("price: min=%.2f p25=%.2f median=%.2f mean=%.2f p75=%.2f max=%.2f" % (
        prices.min(), np.percentile(prices, 25), np.percentile(prices, 50), prices.mean(),
        np.percentile(prices, 75), prices.max()))

    cur.execute("SELECT COUNT(*) FROM Product WHERE SalePrice IS NOT NULL")
    n_sale = cur.fetchone()[0]
    print(f"products on sale (SalePrice not null): {n_sale} / {n_products} ({n_sale/n_products:.1%})")
    cur.execute("SELECT COUNT(*) FROM Product WHERE SalePrice IS NOT NULL AND SalePrice > Price")
    print("SalePrice > Price (inconsistent):", cur.fetchone()[0])
    cur.execute("SELECT DiscountPercentage FROM Product WHERE DiscountPercentage IS NOT NULL")
    discounts = np.array([r[0] for r in cur.fetchall()])
    if len(discounts):
        print("discount%% among discounted: min=%.1f median=%.1f mean=%.1f max=%.1f" % (
            discounts.min(), np.median(discounts), discounts.mean(), discounts.max()))
    cur.execute("""SELECT COUNT(*) FROM Product WHERE SalePrice IS NOT NULL
                   AND ABS(DiscountPercentage - ROUND((Price-SalePrice)/Price*100,1)) > 0.2""")
    print("SalePrice/DiscountPercentage inconsistent rows:", cur.fetchone()[0])

    cur.execute("SELECT COUNT(DISTINCT Brand) FROM Product")
    print("distinct brands:", cur.fetchone()[0])
    cur.execute("SELECT Brand, COUNT(*) FROM Product GROUP BY Brand ORDER BY 2 DESC LIMIT 5")
    print("top brands:", cur.fetchall())
    cur.execute("SELECT COUNT(DISTINCT Name) FROM Product")
    print("distinct names:", cur.fetchone()[0], "/", n_products)
    cur.execute("SELECT COUNT(DISTINCT Description) FROM Product")
    print("distinct descriptions:", cur.fetchone()[0], "/", n_products)
    cur.execute("SELECT COUNT(DISTINCT Slug) FROM Product")
    print("distinct slugs:", cur.fetchone()[0], "/", n_products)
    cur.execute("""SELECT c.Name, COUNT(*) FROM Product p JOIN Category c ON p.CategoryId=c.Id
                   GROUP BY c.Name ORDER BY 2 DESC""")
    print("category distribution:", cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM Product WHERE CategoryId NOT IN (SELECT Id FROM Category)")
    print("dangling CategoryId:", cur.fetchone()[0])

    section("PRODUCT TAGS")
    cur.execute("SELECT COUNT(*) FROM ProductTags")
    print("ProductTags rows:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM ProductTags WHERE ProductId NOT IN (SELECT Id FROM Product)")
    print("dangling ProductId:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM ProductTags WHERE TagId NOT IN (SELECT Id FROM Tag)")
    print("dangling TagId:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM Product WHERE Id NOT IN (SELECT DISTINCT ProductId FROM ProductTags)")
    print("products with zero tags:", cur.fetchone()[0])

    section("USER_EVENTS")
    cur.execute("SELECT COUNT(*) FROM User_events")
    n_events = cur.fetchone()[0]
    print("total events:", n_events)
    for col in ["id", "user_id", "product_id", "action_time", "action_type"]:
        cur.execute(f"SELECT COUNT(*) FROM User_events WHERE {col} IS NULL")
        print(f"null {col}:", cur.fetchone()[0])
    cur.execute("SELECT id, COUNT(*) c FROM User_events GROUP BY id HAVING c>1")
    print("duplicate event ids:", len(cur.fetchall()))
    cur.execute("SELECT user_id, product_id, action_time, action_type, COUNT(*) c FROM User_events GROUP BY user_id, product_id, action_time, action_type HAVING c>1")
    print("exact duplicate event rows:", len(cur.fetchall()))
    cur.execute("SELECT COUNT(*) FROM User_events WHERE user_id NOT IN (SELECT Id FROM User)")
    print("dangling user_id:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM User_events WHERE product_id NOT IN (SELECT Id FROM Product)")
    print("dangling product_id:", cur.fetchone()[0])
    cur.execute("SELECT action_type, COUNT(*) FROM User_events GROUP BY action_type ORDER BY 2 DESC")
    action_counts = cur.fetchall()
    print("action_type counts:", action_counts)
    valid_types = {"CLICK", "ADD_TO_CART", "PURCHASE", "SEARCH", "CHATBOT"}
    invalid = [a for a, _ in action_counts if a not in valid_types]
    print("unexpected action_type values:", invalid)

    cur.execute("SELECT user_id, COUNT(*) c FROM User_events GROUP BY user_id")
    per_user = np.array([r[1] for r in cur.fetchall()])
    print(f"distinct users with events: {len(per_user)} / {n_users}")
    if len(per_user):
        print("events/user: min=%d median=%.1f mean=%.2f p95=%.1f max=%d" % (
            per_user.min(), np.median(per_user), per_user.mean(), np.percentile(per_user, 95), per_user.max()))
    cur.execute("SELECT COUNT(*) FROM User WHERE Id NOT IN (SELECT DISTINCT user_id FROM User_events)")
    n_zero = cur.fetchone()[0]
    print("users with 0 events:", n_zero)
    cur.execute("SELECT user_id, COUNT(*) c FROM User_events GROUP BY user_id HAVING c BETWEEN 1 AND 4")
    print("users with 1-4 events:", len(cur.fetchall()))
    cur.execute("SELECT user_id, COUNT(*) c FROM User_events GROUP BY user_id HAVING c>=5")
    print("users with >=5 events:", len(cur.fetchall()))

    cur.execute("SELECT product_id, COUNT(*) c FROM User_events GROUP BY product_id")
    per_product = np.array([r[1] for r in cur.fetchall()])
    print(f"distinct products with events: {len(per_product)} / {n_products}")
    if len(per_product):
        print("events/product: min=%d median=%.1f mean=%.2f p95=%.1f max=%d" % (
            per_product.min(), np.median(per_product), per_product.mean(), np.percentile(per_product, 95), per_product.max()))

    section("TIMESTAMP READINESS")
    cur.execute("SELECT MIN(action_time), MAX(action_time) FROM User_events")
    earliest, latest = cur.fetchone()
    print("earliest:", earliest, "latest:", latest)
    cur.execute("SELECT COUNT(*) FROM User_events WHERE action_time IS NULL")
    print("null action_time:", cur.fetchone()[0])
    cur.execute("SELECT action_time FROM User_events LIMIT 5")
    print("sample timestamps:", cur.fetchall())

    cur.execute("SELECT user_id, COUNT(DISTINCT action_time) c FROM User_events GROUP BY user_id HAVING c>=2")
    print("users with >=2 distinct timestamped events:", len(cur.fetchall()))
    cur.execute("SELECT user_id, COUNT(DISTINCT action_time) c FROM User_events GROUP BY user_id HAVING c>=5")
    print("users with >=5 distinct timestamped events:", len(cur.fetchall()))
    cur.execute("SELECT user_id, COUNT(DISTINCT action_time) c FROM User_events GROUP BY user_id HAVING c>=10")
    print("users with >=10 distinct timestamped events:", len(cur.fetchall()))

    cur.execute("""SELECT user_id, COUNT(DISTINCT action_time) c FROM User_events
                   WHERE action_type='PURCHASE' GROUP BY user_id HAVING c>=2""")
    print("users with >=2 PURCHASE events at different times:", len(cur.fetchall()))
    cur.execute("""SELECT user_id, COUNT(DISTINCT action_time) c FROM User_events
                   WHERE action_type='PURCHASE' GROUP BY user_id HAVING c>=3""")
    print("users with >=3 PURCHASE events at different times:", len(cur.fetchall()))

    # funnel sanity: within a (user,product) pair, does CLICK usually precede ADD_TO_CART precede PURCHASE?
    cur.execute("""
        SELECT user_id, product_id,
               MIN(CASE WHEN action_type='CLICK' THEN action_time END) as click_t,
               MIN(CASE WHEN action_type='ADD_TO_CART' THEN action_time END) as cart_t,
               MIN(CASE WHEN action_type='PURCHASE' THEN action_time END) as purchase_t
        FROM User_events GROUP BY user_id, product_id
        HAVING cart_t IS NOT NULL AND purchase_t IS NOT NULL
    """)
    rows = cur.fetchall()
    sane = sum(1 for _, _, _, cart_t, purchase_t in rows if cart_t <= purchase_t)
    print(f"cart_time <= purchase_time for pairs with both: {sane}/{len(rows)}")
    cur.execute("""
        SELECT user_id, product_id,
               MIN(CASE WHEN action_type='CLICK' THEN action_time END) as click_t,
               MIN(CASE WHEN action_type='ADD_TO_CART' THEN action_time END) as cart_t
        FROM User_events GROUP BY user_id, product_id
        HAVING click_t IS NOT NULL AND cart_t IS NOT NULL
    """)
    rows2 = cur.fetchall()
    sane2 = sum(1 for _, _, click_t, cart_t in rows2 if click_t <= cart_t)
    print(f"click_time <= cart_time for pairs with both: {sane2}/{len(rows2)}")

    section("PURCHASE GROUND TRUTH")
    cur.execute("SELECT COUNT(*) FROM User_events WHERE action_type='PURCHASE'")
    print("PURCHASE event count:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM User_events WHERE action_type='PURCHASE'")
    print("unique purchasing users:", cur.fetchone()[0])
    cur.execute("""SELECT user_id, COUNT(DISTINCT product_id) c FROM User_events
                   WHERE action_type='PURCHASE' GROUP BY user_id HAVING c>=2""")
    print("users with >=2 distinct purchased products:", len(cur.fetchall()))
    cur.execute("""SELECT user_id, COUNT(DISTINCT product_id) c FROM User_events
                   WHERE action_type='PURCHASE' GROUP BY user_id HAVING c>=3""")
    print("users with >=3 distinct purchased products:", len(cur.fetchall()))
    cur.execute("""SELECT user_id, COUNT(DISTINCT product_id) c FROM User_events
                   WHERE action_type='PURCHASE' GROUP BY user_id HAVING c>=5""")
    print("users with >=5 distinct purchased products:", len(cur.fetchall()))
    cur.execute("""SELECT user_id, product_id, COUNT(*) c FROM User_events
                   WHERE action_type='PURCHASE' GROUP BY user_id, product_id HAVING c>1""")
    repeat_purchases = cur.fetchall()
    print("repeat purchases (same user+product, multiple PURCHASE events):", len(repeat_purchases))

    cur.execute('SELECT COUNT(*) FROM "Order"')
    print("Order rows:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM Order_Item")
    n_oi = cur.fetchone()[0]
    print("Order_Item rows:", n_oi)
    cur.execute('SELECT Status, COUNT(*) FROM "Order" GROUP BY Status')
    print("Order status distribution:", cur.fetchall())
    cur.execute("""SELECT COUNT(*) FROM User_events e WHERE e.action_type='PURCHASE'
                   AND NOT EXISTS (SELECT 1 FROM Order_Item oi JOIN "Order" o ON oi.OrderId=o.Id
                                   WHERE o.UserId=e.user_id AND oi.ProductId=e.product_id)""")
    print("PURCHASE events with NO matching Order_Item (should be 0):", cur.fetchone()[0])
    cur.execute("""SELECT COUNT(*) FROM Order_Item oi JOIN "Order" o ON oi.OrderId=o.Id
                   WHERE o.Status IN ('DELIVERED','COMPLETED','PENDING')
                   AND NOT EXISTS (SELECT 1 FROM User_events e WHERE e.action_type='PURCHASE'
                                   AND e.user_id=o.UserId AND e.product_id=oi.ProductId)""")
    print("Order_Item (non-cancelled) with NO matching PURCHASE event (extra orders, expected >0):", cur.fetchone()[0])

    section("CART")
    cur.execute("SELECT COUNT(*) FROM Cart")
    print("Cart rows:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM Cart_Item")
    print("Cart_Item rows:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(DISTINCT UserId) FROM Cart c JOIN Cart_Item ci ON c.Id=ci.CartId")
    print("users with current non-empty cart:", cur.fetchone()[0])

    section("REVIEWS")
    cur.execute("SELECT COUNT(*) FROM Review")
    print("Review rows:", cur.fetchone()[0])
    cur.execute("SELECT Rating, COUNT(*) FROM Review GROUP BY Rating ORDER BY Rating")
    print("rating distribution:", cur.fetchall())
    cur.execute("SELECT COUNT(DISTINCT UserId) FROM Review")
    print("distinct reviewing users:", cur.fetchone()[0])
    cur.execute("""SELECT COUNT(*) FROM Review r WHERE NOT EXISTS
                   (SELECT 1 FROM User_events e WHERE e.action_type='PURCHASE' AND e.user_id=r.UserId AND e.product_id=r.ProductId)""")
    print("reviews with no corresponding purchase (should be 0):", cur.fetchone()[0])

    section("ELIGIBILITY SUMMARY (for pre-retrieval gate testing)")
    print("active+in-stock (eligible):", active_dist)

    section("VOUCHER / ADDRESS")
    cur.execute("SELECT COUNT(*) FROM Voucher")
    print("Voucher rows:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM UserAddress")
    print("UserAddress rows:", cur.fetchone()[0])
    cur.execute('SELECT COUNT(*) FROM "Order" WHERE VoucherId IS NOT NULL')
    print("orders using a voucher:", cur.fetchone()[0])

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
