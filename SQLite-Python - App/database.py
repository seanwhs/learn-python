# database.py
import sqlite3
import os

path_to_db = './data/customer.db'

# Create database and table
def create_db_table():
    if os.path.exists(path_to_db):
        os.remove(path_to_db)
    conn = sqlite3.connect(path_to_db)

    cursor = conn.cursor()
    create_table = """
        CREATE TABLE IF NOT EXISTS customers (
        first_name TEXT,
        last_name TEXT,
        email_address TEXT
        )
    """
    cursor.execute(create_table)
    conn.commit()
    print('Database anmd Table Created Successfully')
    conn.close()


# Populate Table with a data list
def add_many(customer_list):
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    insert_records = """
        INSERT INTO customers (first_name, last_name, email_address)
        VALUES (?, ?, ?)
    """

    for customer in customer_list:
        cursor.execute(insert_records, customer)

    conn.commit()
    print('Records Inserted Successfully')

# Add a New Record to the Table
def add_one(fname, lname, email):
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    new_record = """ 
        INSERT INTO customers
        VALUES (?, ?, ?)
    """

    cursor.execute(new_record, (fname, lname, email))
    conn.commit()

    print('Record Successfully Added')
    conn.close()

# Delete a Record from the Table
def delete_one(id):
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    delete_record = """ 
        DELETE FROM customers
        WHERE rowid = (?)
    """

    cursor.execute(delete_record, id)

    print('Record Deleted Successfully')
    conn.commit()
    conn.close()

# Query DB and Return All Records
def show_all():
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    select_all = """ 
        SELECT rowid, * 
        FROM customers 
    """

    cursor.execute(select_all)
    results = cursor.fetchall()
    for result in results:
        print (result)
    conn.close()

# Lookup with Where clause
def email_lookup(email):
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    
    match_email = """ 
        SELECT rowid, * 
        FROM customers
        WHERE email_address = (?)
    """

    cursor.execute(match_email, (email,))

    results = cursor.fetchall()
    for result in results:
        print ('\nEmail belongs to:')
        print (result)

    conn.close()
    