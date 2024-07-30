# query.py
import sqlite3

# Path to SQLite database file
path_to_db = './data/customer.db'

# Connect to SQLite database
conn = sqlite3.connect(path_to_db)

# Create a cursor object
cursor = conn.cursor()

# Define the SQL query as a string
# rowid is the primary key SQLite3 created in the background for each record
query = """ 
    SELECT rowid, * FROM customers
"""

# Execute the SQL query
cursor.execute(query)

# Fetch all rows returned by the query
all_rows = cursor.fetchall()

# Iterate over the fetched rows and print each row's data
for row in all_rows:
    # Accessing rowid and other columns (first_name, last_name, email_address)
    print(f"{row[0]}. First Name: {row[1]}, Last Name: {row[2]}, Email: {row[3]}")

# Commit the transaction (though SELECT queries do not modify data, it's good practice)
# This line is typically unnecessary for SELECT queries, but included for completeness
conn.commit()

# Print a success message
print('\nCommand Executed Successfully')

# Close the connection to the database
conn.close()
