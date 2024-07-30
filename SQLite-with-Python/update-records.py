# update-records.py

import sqlite3

# Path to SQLite database file
path_to_db = './data/customer.db'

# Connect to SQLite database
conn = sqlite3.connect(path_to_db)

# Create a cursor object
cursor = conn.cursor()

# Define the SQL query to retrieve customer data before update
query_b4_update = """ 
    SELECT rowid, * 
    FROM customers
    WHERE first_name = 'John'
"""

# Execute the SQL query to retrieve customer data before update
cursor.execute(query_b4_update)

# Fetch all rows returned by the query
results = cursor.fetchall()

# Print customer data before update
print('Before Update:')
for result in results:
    print(result)

# Define the SQL update statement
update = """ 
    UPDATE customers 
    SET last_name = 'Brute', email_address = 'brutal@cruel.com'
    WHERE rowid = 3 AND first_name = 'John'
"""

# Execute the SQL update statement
cursor.execute(update)

# Commit the transaction to save the changes
conn.commit()
print('Update Successful')

# Define the SQL query to retrieve customer data after update
query_after_update = """ 
    SELECT rowid, * 
    FROM customers
    WHERE first_name = 'John'
"""

# Execute the SQL query to retrieve customer data after update
cursor.execute(query_after_update)

# Fetch all rows returned by the query
results = cursor.fetchall()

# Print customer data after update
print('\nAfter Update:')
for result in results:
    print(result)

# Close the connection to the database
conn.close()
