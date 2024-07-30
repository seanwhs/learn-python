# create-database.py
import sqlite3
import os

# Path to SQLite database file
path_to_db = './data/customer.db'

# Check if the database file already exists and remove it if it does
if os.path.exists(path_to_db):
    os.remove(path_to_db)

# Connect to SQLite database
conn = sqlite3.connect(path_to_db)

# Create a cursor object
cursor = conn.cursor()

# Define the CREATE TABLE command as a string
create_table = """
    CREATE TABLE IF NOT EXISTS customers (
        first_name TEXT,
        last_name TEXT,
        email_address TEXT
    )
"""

# Execute the CREATE TABLE command using the cursor
cursor.execute(create_table)

# Commit the transaction
conn.commit()

# Print Success Message
print('Database anmd Table Created Successfully')

# Close the connection
conn.close()
