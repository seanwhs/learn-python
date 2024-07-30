import numpy as np
import pandas as pd

def gradient_descent(x, y):
    m_current = b_current = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.0002
    
    for i in range(iterations):
        y_predicted = m_current * x + b_current
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        m_derivative = -(2/n) * sum(x * (y - y_predicted))
        b_derivative = -(2/n) * sum(y - y_predicted)
        m_current = m_current - (learning_rate * m_derivative)
        b_current = b_current - (learning_rate * b_derivative)
        print(f'Iteration {i+1}: m = {m_current}, b = {b_current}, cost = {cost}')

# Load data from CSV file
df = pd.read_csv('./data/test_scores.csv')

# Convert columns to NumPy arrays
x = df['math'].to_numpy()
y = df['cs'].to_numpy()

# Call gradient_descent function with loaded data
gradient_descent(x, y)
