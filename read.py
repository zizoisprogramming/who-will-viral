import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & PROCESS DIGINETICA
# ══════════════════════════════════════════════════════════════
print("Loading Diginetica...")

digi_views     = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Diginetica/train-item-views.csv", sep=";")
digi_purchases = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Diginetica/train-purchases.csv",  sep=";")
digi_products  = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Diginetica/products.csv",         sep=";")
digi_cats      = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Diginetica/product-categories.csv", sep=";")

# Normalize column names
digi_views     = digi_views.rename(columns={"sessionId": "session_id", "itemId": "item_id", "timeframe": "timeframe", "eventdate": "eventdate"})
digi_purchases = digi_purchases.rename(columns={"sessionId": "session_id", "itemId": "item_id", "timeframe": "timeframe", "eventdate": "eventdate"})

digi_views["is_buy"]     = 0
digi_purchases["is_buy"] = 1

digi = pd.concat([digi_views, digi_purchases], ignore_index=True)
digi = digi.sort_values(["session_id", "timeframe"]).reset_index(drop=True)

# Join price from products.csv
if "itemId" in digi_products.columns:
    digi_products = digi_products.rename(columns={"itemId": "item_id", "pricelog2": "price"})
    digi = digi.merge(digi_products[["item_id", "price"]], on="item_id", how="left")
else:
    digi["price"] = np.nan

# Join category from product-categories.csv
if "itemId" in digi_cats.columns:
    digi_cats = digi_cats.rename(columns={"itemId": "item_id", "categoryId": "category_id"})
    digi = digi.merge(digi_cats[["item_id", "category_id"]], on="item_id", how="left")
else:
    digi["category_id"] = np.nan

# Temporal features
digi["eventdate"] = pd.to_datetime(digi["eventdate"], errors="coerce")
digi["hour"]        = digi["eventdate"].dt.hour
digi["day_of_week"] = digi["eventdate"].dt.dayofweek
digi["is_weekend"]  = digi["day_of_week"].isin([5, 6]).astype(int)
digi["month"]       = digi["eventdate"].dt.month

# Re-index item & session IDs with a dataset prefix to avoid collisions
digi["item_id"]    = "D_" + digi["item_id"].astype(str)
digi["session_id"] = "D_" + digi["session_id"].astype(str)
digi["user_id"]    = digi["session_id"]   # Diginetica has no user_id, use session
digi["dataset"]    = "diginetica"

# Keep unified schema columns
digi_unified = digi[[
    "dataset", "user_id", "session_id", "item_id",
    "is_buy", "price", "category_id",
    "hour", "day_of_week", "is_weekend", "month"
]]

print(f"  Diginetica rows: {len(digi_unified):,}")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — LOAD & PROCESS RETAILROCKET
# ══════════════════════════════════════════════════════════════
print("Loading RetailRocket...")

rr_events = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Retail/events.csv")
rr_props1 = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Retail/item_properties_part1.csv")
rr_props2 = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Retail/item_properties_part2.csv")
rr_cats   = pd.read_csv("/home/zizo/MyHouse/CMP Year 4/Semester 2/Data Science/Project/dataset/Retail/category_tree.csv")

# is_buy flag
rr_events["is_buy"] = (rr_events["event"] == "transaction").astype(int)

# Datetime features
rr_events["datetime"]    = pd.to_datetime(rr_events["timestamp"], unit="ms")
rr_events["hour"]        = rr_events["datetime"].dt.hour
rr_events["day_of_week"] = rr_events["datetime"].dt.dayofweek
rr_events["is_weekend"]  = rr_events["day_of_week"].isin([5, 6]).astype(int)
rr_events["month"]       = rr_events["datetime"].dt.month

# Extract category & price from item properties
rr_props = pd.concat([rr_props1, rr_props2], ignore_index=True)
rr_props_pivot = (
    rr_props[rr_props["property"].isin(["categoryid", "790"])]
    .pivot_table(index="itemid", columns="property", values="value", aggfunc="last")
    .reset_index()
)
rr_props_pivot.columns.name = None
rr_props_pivot = rr_props_pivot.rename(columns={
    "itemid":     "item_id",
    "categoryid": "category_id",
    "790":        "price"           # property 790 is commonly price
})

rr_events = rr_events.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
rr_events = rr_events.merge(rr_props_pivot, on="item_id", how="left")

# Assign session_id — RetailRocket has no explicit sessions,
# so we create them: new session if gap > 30 minutes
rr_events = rr_events.sort_values(["user_id", "timestamp"])
rr_events["time_diff"] = rr_events.groupby("user_id")["timestamp"].diff().fillna(0)
rr_events["new_session"] = (rr_events["time_diff"] > 30 * 60 * 1000).astype(int)
rr_events["session_id"] = (
    "RR_" +
    rr_events["user_id"].astype(str) + "_" +
    rr_events.groupby("user_id")["new_session"].cumsum().astype(str)
)

# Prefix IDs to avoid collision with Diginetica
rr_events["item_id"] = "RR_" + rr_events["item_id"].astype(str)
rr_events["user_id"] = "RR_" + rr_events["user_id"].astype(str)
rr_events["dataset"] = "retailrocket"

rr_unified = rr_events[[
    "dataset", "user_id", "session_id", "item_id",
    "is_buy", "price", "category_id",
    "hour", "day_of_week", "is_weekend", "month"
]]

print(f"  RetailRocket rows: {len(rr_unified):,}")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — STACK INTO ONE UNIFIED DATASET
# ══════════════════════════════════════════════════════════════
print("Combining datasets...")

unified = pd.concat([digi_unified, rr_unified], ignore_index=True)

# ── Drop sessions with < 3 interactions (standard practice) ──
session_counts = unified.groupby("session_id")["item_id"].count()
valid_sessions = session_counts[session_counts >= 3].index
unified = unified[unified["session_id"].isin(valid_sessions)].reset_index(drop=True)

# ── Encode item_id & session_id as integers for model input ──
unified["item_id_enc"]    = unified["item_id"].astype("category").cat.codes
unified["session_id_enc"] = unified["session_id"].astype("category").cat.codes
unified["user_id_enc"]    = unified["user_id"].astype("category").cat.codes

# ── Normalize price ───────────────────────────────────────────
unified["price"] = pd.to_numeric(unified["price"], errors="coerce")
unified["price_norm"] = (
    (unified["price"] - unified["price"].min()) /
    (unified["price"].max() - unified["price"].min())
)

# ── Item popularity features (computed across full corpus) ────
item_stats = unified.groupby("item_id_enc").agg(
    item_total_interactions=("is_buy", "count"),
    item_purchase_count=("is_buy", "sum"),
).reset_index()
item_stats["item_conversion_rate"] = (
    item_stats["item_purchase_count"] / item_stats["item_total_interactions"]
)
item_stats["item_popularity_rank"] = (
    item_stats["item_total_interactions"].rank(ascending=False).astype(int)
)
unified = unified.merge(item_stats, on="item_id_enc", how="left")

# ── Session-level features ────────────────────────────────────
session_stats = unified.groupby("session_id_enc").agg(
    session_length=("item_id_enc", "count"),
    num_unique_items=("item_id_enc", "nunique"),
    session_purchase_count=("is_buy", "sum"),
).reset_index()
session_stats["session_purchase_rate"] = (
    session_stats["session_purchase_count"] / session_stats["session_length"]
)
unified = unified.merge(session_stats, on="session_id_enc", how="left")

# ── Item position in session ──────────────────────────────────
unified["item_position"] = unified.groupby("session_id_enc").cumcount() + 1


# ══════════════════════════════════════════════════════════════
#  STEP 4 — FINAL SCHEMA & SAVE
# ══════════════════════════════════════════════════════════════
final_cols = [
    # Identifiers
    "dataset", "user_id_enc", "session_id_enc", "item_id_enc",
    # Labels
    "is_buy",
    # Item features
    "price_norm", "category_id",
    "item_total_interactions", "item_purchase_count",
    "item_conversion_rate", "item_popularity_rank",
    # Session features
    "session_length", "num_unique_items",
    "session_purchase_count", "session_purchase_rate",
    "item_position",
    # Temporal features
    "hour", "day_of_week", "is_weekend", "month",
    # Source strings (keep for debugging)
    "user_id", "session_id", "item_id",
]

unified = unified[final_cols]
unified.to_csv("unified_dataset.csv", index=False)

print("\n✅ Done!")
print(f"   Total rows      : {len(unified):,}")
print(f"   Unique sessions : {unified['session_id_enc'].nunique():,}")
print(f"   Unique items    : {unified['item_id_enc'].nunique():,}")
print(f"   Diginetica rows : {(unified['dataset']=='diginetica').sum():,}")
print(f"   RetailRocket rows: {(unified['dataset']=='retailrocket').sum():,}")
print(f"   Purchase rate   : {unified['is_buy'].mean():.4f}")
print(f"\n   Columns: {unified.columns.tolist()}")