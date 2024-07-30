# my-app.py
import database
from data.customer_data import customers # Importing from ./data/customer_data.py

database.create_db_table()

database.add_many(customers)
print('\nAfter Insertion:')
database.show_all()

database.add_one('Jerry', 'Louise', 'jerry@new-mail.com')
print('\nAfter Adding One Record:')
database.show_all()

database.delete_one('9')
print('\nAfter Deleting One Record:')
database.show_all()

database.email_lookup('sean@email.com')
