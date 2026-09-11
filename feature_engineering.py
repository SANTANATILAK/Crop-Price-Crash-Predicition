import os
import pandas as pd
import numpy as np
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

connection = mysql.connector.connect(
    host=os.environ.get("DB_HOST", "localhost"),
    user=os.environ.get("DB_USER", "root"),
    password=os.environ.get("DB_PASSWORD"),
    database=os.environ.get("DB_NAME", "crop_price_predictor")
)

query = """
SELECT
    m.market_name,
    m.district_name,
    m.state,
    c.commodity,
    c.variety,
    c.grade,
    p.price_date,
    p.modal_price
FROM price_records p
JOIN mandis m ON p.mandi_id = m.mandi_id
JOIN crops c ON p.crop_id = c.crop_id
WHERE c.commodity IN (
    'Banana',
    'Brinjal',
    'Cabbage',
    'Garlic',
    'Green Chilli'
)
ORDER BY m.market_name, c.commodity, p.price_date
"""

df = pd.read_sql(query, connection)

connection.close()

print("Rows loaded:", len(df))

if df.empty:
    print("No data found.")
    exit()

df["price_date"] = pd.to_datetime(df["price_date"])
df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")

df = df.dropna(subset=["modal_price", "price_date"])
df = df[df["modal_price"] > 0]

df["commodity"] = df["commodity"].str.strip()
df["variety"] = df["variety"].str.strip()
df["grade"] = df["grade"].str.strip()
df["market_name"] = df["market_name"].str.strip()
df["district_name"] = df["district_name"].str.strip()
df["state"] = df["state"].str.strip()

group_columns = [
    "market_name",
    "district_name",
    "state",
    "commodity",
    "variety",
    "grade"
]

result = []

for _, group in df.groupby(group_columns, sort=False):

    group = group.sort_values("price_date").copy()

    group["price_lag_1"] = group["modal_price"].shift(1)
    group["price_lag_3"] = group["modal_price"].shift(3)
    group["price_lag_7"] = group["modal_price"].shift(7)
    group["price_lag_14"] = group["modal_price"].shift(14)
    group["price_lag_30"] = group["modal_price"].shift(30)

    group["price_change_3d"] = (
        group["modal_price"] / group["price_lag_3"] - 1
    )

    group["price_change_7d"] = (
        group["modal_price"] / group["price_lag_7"] - 1
    )

    group["price_change_14d"] = (
        group["modal_price"] / group["price_lag_14"] - 1
    )

    group["price_change_30d"] = (
        group["modal_price"] / group["price_lag_30"] - 1
    )

    price_series = group.set_index("price_date")["modal_price"]

    group["rolling_mean_7d"] = (
        price_series.rolling("7D", min_periods=3).mean().values
    )

    group["rolling_mean_30d"] = (
        price_series.rolling("30D", min_periods=7).mean().values
    )

    group["rolling_std_7d"] = (
        price_series.rolling("7D", min_periods=3).std().values
    )

    group["rolling_std_30d"] = (
        price_series.rolling("30D", min_periods=7).std().values
    )

    group["price_vs_ma7"] = (
        group["modal_price"] / group["rolling_mean_7d"] - 1
    )

    group["price_vs_ma30"] = (
        group["modal_price"] / group["rolling_mean_30d"] - 1
    )

    group["month"] = group["price_date"].dt.month
    group["day_of_week"] = group["price_date"].dt.dayofweek

    reverse = group[["price_date", "modal_price"]].copy()

    reverse["_reverse_date"] = (
        reverse["price_date"].max() - reverse["price_date"]
    )

    reverse = reverse.sort_values("_reverse_date")

    reverse["future_min_7d"] = (
        reverse.set_index("_reverse_date")["modal_price"]
        .shift(1)
        .rolling("7D", closed="both", min_periods=1)
        .min()
        .values
    )

    reverse = reverse.sort_values("price_date")

    group["future_min_7d"] = reverse["future_min_7d"].values

    group["future_drop_pct"] = (
        group["future_min_7d"] / group["modal_price"] - 1
    )

    group["crash"] = np.where(
        group["future_min_7d"].isna(),
        np.nan,
        (group["future_drop_pct"] <= -0.15).astype(int)
    )

    result.append(group)

if len(result) == 0:
    print("No groups were created.")
    exit()

ml_df = pd.concat(result, ignore_index=True)

ml_df = ml_df.dropna(
    subset=[
        "price_lag_1",
        "price_lag_3",
        "price_lag_7",
        "price_lag_14",
        "price_lag_30",
        "price_change_3d",
        "price_change_7d",
        "price_change_14d",
        "price_change_30d",
        "rolling_mean_7d",
        "rolling_mean_30d",
        "rolling_std_7d",
        "rolling_std_30d",
        "price_vs_ma7",
        "price_vs_ma30",
        "crash"
    ]
)

ml_df["crash"] = ml_df["crash"].astype(int)

output_columns = [
    "market_name",
    "district_name",
    "state",
    "commodity",
    "variety",
    "grade",
    "price_date",
    "modal_price",
    "price_lag_1",
    "price_lag_3",
    "price_lag_7",
    "price_lag_14",
    "price_lag_30",
    "price_change_3d",
    "price_change_7d",
    "price_change_14d",
    "price_change_30d",
    "rolling_mean_7d",
    "rolling_mean_30d",
    "rolling_std_7d",
    "rolling_std_30d",
    "price_vs_ma7",
    "price_vs_ma30",
    "month",
    "day_of_week",
    "crash"
]

ml_df = ml_df[output_columns]

ml_df.to_csv("crop_price_ml.csv", index=False)

print("\nFeature engineering completed!")
print("Rows:", len(ml_df))
print("Columns:", len(ml_df.columns))

print("\nCrop distribution:")
print(ml_df["commodity"].value_counts())

print("\nCrash distribution:")
print(ml_df["crash"].value_counts())

print("\nCrash percentage:")
print((ml_df["crash"].value_counts(normalize=True) * 100).round(2))

print("\nSample:")
print(ml_df.head())

print("\nOutput file created: crop_price_ml.csv")