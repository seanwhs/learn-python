# query-where-clause.py

import sqlite3

# Path to SQLite database file
path_to_db = './data/customer.db'

# Connect to SQLite database
conn = sqlite3.connect(path_to_db)

# Create a cursor object
cursor = conn.cursor()

# Define the SQL query to retrieve customer data by primary key (rowid = 5)
query_by_pk = """ 
    SELECT * 
    FROM customers 
    WHERE rowid = 5
"""

# Define the SQL query to retrieve email addresses by last name ('Wong')
query_email_by_last_name = """ 
    SELECT email_address 
    FROM customers 
    WHERE last_name = 'Wong'
"""

# Define the SQL query to retrieve customer data by first name starting with 'Jo'
query_email_by_first_name = """ 
    SELECT first_name, last_name, email_address 
    FROM customers 
    WHERE first_name LIKE 'Jo%'
"""

# Define the SQL query to retrieve customer data by first name starting with 'Jo' 
# and rowid less than 4
query_email_by_first_name_and_rowid = """ 
    SELECT first_name, last_name, email_address 
    FROM customers 
    WHERE (first_name LIKE 'Jo%' AND rowid < 4)
"""

# Define the SQL query to retrieve customer data by email address domain ending with 'qmail.com'
query_by_email = """ 
    SELECT first_name, last_name, email_address 
    FROM customers 
    WHERE email_address LIKE '%qmail.com'
"""

# Choose query to test (uncomment one)
cursor.execute(query_by_email)
# cursor.execute(query_by_pk)
# cursor.execute(query_email_by_last_name)
# cursor.execute(query_email_by_first_name)
# cursor.execute(query_email_by_first_name_and_rowid)

# Fetch all rows returned by the query
results = cursor.fetchall()

# Iterate over the fetched results and print each result
for result in results:
    print(result)

# Close the connection to the database
conn.close()
