#myFunctions.py
from datetime import datetime

def greeting():
    firstname = input('Enter firstname: ')
    lastname = input('Enter lastname: ')
    print(f'Hello {firstname} {lastname}!')

def time_check():
    now = datetime.now()
    day = now.day
    month = now.month
    year = now.year
    print(f"Today's date  is {day}/{month}/{year}")
    print(f"Time now is {now.hour}:{now.minute}:{now.second}")

def calculate_average(numList):
    total = 0
    for num in numList:
        total += num
    average = total/len(numList)
    return average