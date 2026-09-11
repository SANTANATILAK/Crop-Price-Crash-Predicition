import os
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

cursor.execute("""
SELECT DISTINCT commodity
FROM crops
ORDER BY commodity
""")

rows = cursor.fetchall()

for row in rows:
    print(row[0])

cursor.close()
connection.close()