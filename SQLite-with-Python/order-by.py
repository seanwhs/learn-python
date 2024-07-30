#order-by.py
import sqlite3

path_to_db = './data/customer.db'
conn = sqlite3.connect(path_to_db)

cursor = conn.cursor()

select_all = """ 
    SELECT rowid, * 
    FROM customers
"""

order_by_asc = """ 
    SELECT rowid, * 
    FROM customers
    ORDER BY last_name
"""

order_by_dsc = """ 
    SELECT rowid, * 
    FROM customers
    ORDER BY last_name DESC
"""

def print_listing():
    results = cursor.fetchall()
    for result in results:
        print(result)

cursor.execute(select_all)
print('Unordered Listing:')
print_listing()

cursor.execute(order_by_asc)
print('Ascending Order:')
print_listing()

cursor.execute(order_by_dsc)
print('Descending:')
print_listing()




