# Author: Karen Rugerio
# Linear regression using scikit learn
# Dataset: Correlation between years of experience and salary. Source: https://github.com/mohit-baliyan/references/blob/master/salary_data.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def plot_results(x: np.ndarray, y: np.ndarray, prediction: np.ndarray)->None:
    """
    Function to scatter the samples and the returned function
    to be ploted
    """
    plt.scatter(x, y,  color='black')
    plt.plot(x, prediction, color='#F5451F')
    plt.legend(['POINTS', 'FITTED FUNCTION'])
    plt.show()


if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv('salary_data.csv')
    # Calculate number of elements present on the data frame
    elements = len(df)
    x = df.YearsExperience.values
    y = df.Salary.values

    # Estimated coefficients for the linear regression problem
    # this is a 1D array of length n_features
    x = x.reshape(elements, 1)
    y = y.reshape(elements, 1)

    # Apply the linear regression by Sklearn library
    regr = linear_model.LinearRegression()
    #Fit linear model
    regr.fit(x, y)

    #Predict using the linear model.
    prediction = regr.predict(x)

    plot_results(x,y, prediction)

