# gradient_descent.py

# Importing necessary library
import numpy as np

# Function to perform gradient descent for linear regression
def gradient_descent(x, y):
    m_current = b_current = 0  # Initial values for slope (m) and intercept (b)
    iterations = 10000  # Number of iterations for gradient descent
    n = len(x)  # Number of data points
    learning_rate = 0.001  # Learning rate for gradient descent
    
    # Iterating through the gradient descent process
    for i in range(iterations):
        y_predicted = (m_current * x) + b_current  # Current predicted values of y
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])  # Cost function (mean squared error)
        
        # Partial derivatives of cost function with respect to m and b
        m_derivative = -(2/n) * sum(x * (y - y_predicted))
        b_derivative = -(2/n) * sum(y - y_predicted)
        
        # Update m and b using gradient descent update rule
        m_current = m_current - (learning_rate * m_derivative)
        b_current = b_current - (learning_rate * b_derivative)
        
        # Print progress of gradient descent
        print(f'Iteration {i+1}: m = {m_current}, b = {b_current}, cost = {cost}')

# Input data (x) and output data (y)
x = np.array([1, 2, 3, 4, 5])  # Input vector (independent variable)
y = np.array([5, 7, 9, 11, 13])  # Output vector (dependent variable)

# Call gradient_descent function with the provided data
gradient_descent(x, y)
