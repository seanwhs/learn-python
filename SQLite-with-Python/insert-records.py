# insert-records.py

import sqlite3

# Path to SQLite database file
path_to_database = './data/customer.db'

# List of customer records to insert
customers = [
    ('Sean', 'Wong', 'sean@email.com'), 
    ('Travis', 'Wong', 'travis@email.com'), 
    ('John', 'Travolta', 'john@email.com'), 
    ('John', 'Trump', 'john-trump@qmail.com'), 
    ('Tim', 'Blake', 'tim@email.com'),
    ('James', 'Brown', 'james@email.com'),
    ('James', 'Seymour', 'james-seymour@qmail.com'),
    ('Joe', 'Biden', 'joe@email.com')
]

# SQL command to insert records into customers table
insert_records = """
    INSERT INTO customers (first_name, last_name, email_address)
    VALUES (?, ?, ?)
"""

# Connect to SQLite database
conn = sqlite3.connect(path_to_database)

# Create a cursor object
cursor = conn.cursor()

# Clear existing data in the customers table (for testing purposes)
cursor.execute('DELETE FROM customers')

# Execute the SQL command for each customer record
for customer in customers:
    cursor.execute(insert_records, customer)

# Commit the transaction to save changes
conn.commit()

# Print Success Message
print('Records Inserted Successfully')

# Close the database connection
conn.close()
