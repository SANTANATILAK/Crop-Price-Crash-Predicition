import os
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

connection = mysql.connector.connect(
    host=os.environ.get("DB_HOST", "localhost"),
    user=os.environ.get("DB_USER", "root"),
    password=os.environ.get("DB_PASSWORD"),
    database=os.environ.get("DB_NAME", "crop_price_predictor")
)

print("MySQL connected successfully!")

query = """
SELECT
    p.price_date,
    p.modal_price,
    p.min_price,
    p.max_price,
    c.commodity,
    c.variety,
    c.grade,
    m.market_name,
    m.district_name,
    m.state
FROM price_records p
JOIN crops c
    ON p.crop_id = c.crop_id
JOIN mandis m
    ON p.mandi_id = m.mandi_id
"""

df = pd.read_sql(query, connection)

connection.close()

print("\nData loaded successfully!")
print("Total records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 records:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

commodity_counts = df["commodity"].value_counts()

print("\nCommodities:")
print(commodity_counts)

print("\nStates:")
print(df["state"].value_counts())

print("\nPrice statistics:")
print(df[["min_price", "max_price", "modal_price"]].describe())

plt.figure(figsize=(12, 6))
commodity_counts.head(10).plot(kind="bar")
plt.title("Top 10 Commodities by Number of Records")
plt.xlabel("Commodity")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

tomato = df[df["commodity"].str.lower() == "tomato"].copy()

if len(tomato) > 0:
    market = tomato["market_name"].value_counts().index[0]

    tomato_market = tomato[tomato["market_name"] == market].copy()
    tomato_market = tomato_market.sort_values("price_date")

    plt.figure(figsize=(14, 6))
    plt.plot(
        tomato_market["price_date"],
        tomato_market["modal_price"]
    )
    plt.title(f"Tomato Modal Price Trend - {market}")
    plt.xlabel("Date")
    plt.ylabel("Modal Price (Rs./Quintal)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nTomato market:", market)
else:
    print("\nTomato data not found.")

onion = df[df["commodity"].str.lower() == "onion"].copy()

if len(onion) > 0:
    market = onion["market_name"].value_counts().index[0]

    onion_market = onion[onion["market_name"] == market].copy()
    onion_market = onion_market.sort_values("price_date")

    plt.figure(figsize=(14, 6))
    plt.plot(
        onion_market["price_date"],
        onion_market["modal_price"]
    )
    plt.title(f"Onion Modal Price Trend - {market}")
    plt.xlabel("Date")
    plt.ylabel("Modal Price (Rs./Quintal)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nOnion market:", market)
else:
    print("\nOnion data not found.")

potato = df[df["commodity"].str.lower() == "potato"].copy()

if len(potato) > 0:
    market = potato["market_name"].value_counts().index[0]

    potato_market = potato[potato["market_name"] == market].copy()
    potato_market = potato_market.sort_values("price_date")

    plt.figure(figsize=(14, 6))
    plt.plot(
        potato_market["price_date"],
        potato_market["modal_price"]
    )
    plt.title(f"Potato Modal Price Trend - {market}")
    plt.xlabel("Date")
    plt.ylabel("Modal Price (Rs./Quintal)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nPotato market:", market)
else:
    print("\nPotato data not found.")

print("\nEDA completed successfully!")