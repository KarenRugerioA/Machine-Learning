# Linear regeression using gradient descent optimization
# Author Karen Rugerio
# Dataset: Correlation between years of experience and salary 

from ctypes import sizeof
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def gradient_descent(actual_m: float, actual_b: float, points: pd.DataFrame, learning_rate: float, elements: int)->Tuple[float,float]:
    
    """
    Function to calculate the gradient descent
    given an actual m and b,  a Data Fram with x and y values
    and the learning rate to be used
    """
    # Partial derivatives initialization 
    m_gradient = 0
    b_gradient = 0

    

    for i in range(elements):
        # Get x and y values
        x = points.iloc[i].YearsExperience 
        y = points.iloc[i].Salary

        # Calculate partial derivative respect to m 
        m_gradient += -(2/elements) * x * (y - (actual_m * x + actual_b))

        # Calculate partial derivative respect to b. 
        b_gradient += -(2/elements) * (y - (actual_m * x + actual_b))

    #Calculate the resulting m and b using the results of the partial derivatives
    m = actual_m - m_gradient * learning_rate
    b = actual_b - b_gradient * learning_rate

    return m, b

def plot_results(df: pd.DataFrame, m: float, b: float)->None:
    
    """
    Function to scatter the samples and the returned function
    to be ploted
    """
    plt.scatter(df.YearsExperience, df.Salary)
    plt.plot(list(range(0,12)), [m * x + b for x in range (0,12)], color="#F5451F")
    plt.legend(['FITTED FUNCTION', 'POINTS'])
    plt.show()

if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv('salary_data.csv')
    
    # Initialize actual m and actual b 
    m = 0.0 
    b = 0.0
    
    # Define a learning rate and epochs
    learning_rate = 0.001 
    epochs = 500
    
    # Calculate number of elements present on the data frame
    elements = len(df)
    
    for i in range(epochs):
        print(f"Processed epoch: {i}\n m: {m}\n b: {b}")
        # Calculate current gradient descent
        m, b = gradient_descent(m, b, df, learning_rate, elements) 

    plot_results(df, m, b)