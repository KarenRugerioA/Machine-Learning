# Author: Karen Rugerio
# Linear regression using scikit learn
# Dataset: Correlation between years of experience and salary. Source: https://github.com/mohit-baliyan/references/blob/master/salary_data.csv

from cgitb import text
import sys
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def plot_results(x: np.ndarray, y: np.ndarray, prediction: np.ndarray, t_prediction: Tuple)->None:
    """
    Function to scatter the samples and the returned function
    to be ploted
    """
    # t_prediction[0] is the given years of experience
    # t_prediction[1] is the predicted salary at the given years of experience
    new_x = np.append(x, t_prediction[0])
    new_prediction = np.append(prediction, t_prediction[1])

    plt.scatter(x, y,  color='black')
    plt.scatter(t_prediction[0], t_prediction[1], color='blue')
    plt.annotate('Predicted value', xy=t_prediction, horizontalalignment='right', verticalalignment='top')
    plt.plot(new_x, new_prediction, color='#F5451F')
    plt.legend(['POINTS', 'PREDICTED VALUE','FITTED FUNCTION'])
    plt.show()


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Linear Regression using Scikit learn', usage='linearRegressionSci.py -p [# Years of experience to predict Salary of]')
    parser.add_argument('-p', '--prediction', type=int, help='Years of experience to predict Salary of', required=True)
    args = parser.parse_args()
    
    # Validate the value to predict is > than 0
    if (args.prediction <= 0):
        sys.exit('Cannot predict 0 or negative years of experience')

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
    y_prediction = regr.predict(x)
    print(f"Predicted values:\n {y_prediction}")

    # Predict Salary at a given value
    predicted_at_value = regr.predict([[args.prediction]])
    print(f"Prediction at {args.prediction}: {predicted_at_value}")

    # Create a tuple with the given years of experience (x) and the predicted
    # salary for that given value (y)
    t_prediction = (args.prediction, predicted_at_value)

    plot_results(x,y, y_prediction, t_prediction)

