import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

connection = mysql.connector.connect(
    host=os.environ.get("DB_HOST", "localhost"),
    user=os.environ.get("DB_USER", "root"),
    password=os.environ.get("DB_PASSWORD"),
    database=os.environ.get("DB_NAME", "crop_price_predictor")
)

cursor = connection.cursor()

print("MySQL connected successfully!")


csv_file = "agmarknet_india_historical_prices_2024_2025.csv"

chunk_size = 10000

print("\nLoading mandis and crops...")

for chunk in pd.read_csv(csv_file, chunksize=chunk_size):

    chunk.columns = chunk.columns.str.strip()

    mandi_data = chunk[
        ["Market Name", "District Name", "State"]
    ].drop_duplicates()

    for _, row in mandi_data.iterrows():

        sql = """
        INSERT IGNORE INTO mandis
        (market_name, district_name, state)
        VALUES (%s, %s, %s)
        """

        values = (
            row["Market Name"],
            row["District Name"],
            row["State"]
        )

        cursor.execute(sql, values)

    crop_data = chunk[
        ["Commodity", "Variety", "Grade"]
    ].drop_duplicates()

    for _, row in crop_data.iterrows():

        sql = """
        INSERT IGNORE INTO crops
        (commodity, variety, grade)
        VALUES (%s, %s, %s)
        """

        values = (
            row["Commodity"],
            row["Variety"],
            row["Grade"]
        )

        cursor.execute(sql, values)

    connection.commit()

print("Mandis and crops loaded successfully!")


print("\nGetting IDs from MySQL...")


cursor.execute("""
    SELECT mandi_id, market_name, district_name, state
    FROM mandis
""")

mandi_lookup = {}

for row in cursor.fetchall():

    mandi_id = row[0]
    market_name = row[1]
    district_name = row[2]
    state = row[3]

    key = (
        market_name,
        district_name,
        state
    )

    mandi_lookup[key] = mandi_id

cursor.execute("""
    SELECT crop_id, commodity, variety, grade
    FROM crops
""")

crop_lookup = {}

for row in cursor.fetchall():

    crop_id = row[0]
    commodity = row[1]
    variety = row[2]
    grade = row[3]

    key = (
        commodity,
        variety,
        grade
    )

    crop_lookup[key] = crop_id


print("Number of mandis:", len(mandi_lookup))
print("Number of crops:", len(crop_lookup))

print("\nLoading price records...")

total_rows = 0

for chunk in pd.read_csv(
    csv_file,
    chunksize=chunk_size
):

    chunk.columns = chunk.columns.str.strip()
    chunk["Price Date"] = pd.to_datetime(chunk["Price Date"], format="%d %b %Y").dt.strftime("%Y-%m-%d")

    price_values = []

    for _, row in chunk.iterrows():

        mandi_key = (
            row["Market Name"],
            row["District Name"],
            row["State"]
        )

        crop_key = (
            row["Commodity"],
            row["Variety"],
            row["Grade"]
        )

        mandi_id = mandi_lookup.get(mandi_key)
        crop_id = crop_lookup.get(crop_key)

        if mandi_id is None:
            continue

        if crop_id is None:
            continue

        price_values.append(
            (
                mandi_id,
                crop_id,
                row["Price Date"],
                row["Min Price (Rs./Quintal)"],
                row["Max Price (Rs./Quintal)"],
                row["Modal Price (Rs./Quintal)"]
            )
        )

    if len(price_values) > 0:

        cursor.executemany(
            """
            INSERT INTO price_records
            (
                mandi_id,
                crop_id,
                price_date,
                min_price,
                max_price,
                modal_price
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            price_values
        )

        connection.commit()

        total_rows += len(price_values)

        print(
            "Price records loaded:",
            total_rows
        )

cursor.close()
connection.close()
print("DATA LOADING COMPLETED")
print("Total price records:", total_rows)