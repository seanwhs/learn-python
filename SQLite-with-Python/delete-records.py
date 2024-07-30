# delete-records.py

import sqlite3

# Path to SQLite database file
path_to_db = './data/customer.db'

# Connect to SQLite database
conn = sqlite3.connect(path_to_db)

# Create a cursor object
cursor = conn.cursor()

# Function to print all records in the customers table
def print_all_records():
    print_records = """ 
        SELECT rowid, * FROM customers
    """
    cursor.execute(print_records)
    results = cursor.fetchall()
    for result in results:
        print(result)

# Print existing records before deletion
print('Before Deleting:')
print_all_records()

# Function to delete one specific record from the customers table
def delete_one_record():
    delete_record = """ 
        DELETE 
        FROM customers 
        WHERE rowid = 8 
    """
    cursor.execute(delete_record)
    conn.commit()

# Execute the function to delete a specific record
delete_one_record()

# Print records after deletion
print('After Deleting:')
print_all_records()

# Close the connection to the database
conn.close()
