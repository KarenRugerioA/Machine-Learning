# Author: Karen Rugerio
# Linear regression using scikit learn
"""
Dataset:
Swedish Committee on Analysis of Risk Premium in Motor Insurance
http://college.hmco.com/mathematics/brase/understandable_statistics/7e/students/datasets/
       slr/frames/frame.htmlownlee/Datasets/blob/master/auto-insurance.csv
x = number of claims.
y = total payment for all the claims in thousands Swedish Kronor.
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn import linear_model

def plot_results(x: np.ndarray, y: np.ndarray, y_prediction: np.ndarray, t_prediction: Tuple)->None:
    """
    Function to scatter the samples and the returned function
    to be ploted
    """
    # t_prediction[0] is the number of claims
    # t_prediction[1] is the total payment for all the claims in thousands
    new_x = np.append(x, t_prediction[0])
    new_y_prediction = np.append(y_prediction, t_prediction[1])

    plt.scatter(x, y,  color='#626361')
    plt.scatter(t_prediction[0], t_prediction[1], color='#050df7')
    plt.annotate('Predicted value', xy=t_prediction, horizontalalignment='right', verticalalignment='top')
    plt.plot(new_x, new_y_prediction, color='#F5451F')
    plt.legend(['POINTS', 'PREDICTED VALUE','FITTED FUNCTION'])
    plt.show()


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Linear Regression using Scikit learn', usage='python3 linearRegressionSci.py -p [# numbers of claims to predict total payment of]')
    parser.add_argument('-p', '--prediction', type=float, help='Numbers of claims to predict total payment of', required=True)
    args = parser.parse_args() #Saves parser arguments
    
    # Validate the value to predict is > than 0
    if (args.prediction <= 0):
        sys.exit('Cannot predict 0 or negative number of claims')

    # Read the dataset
    df = pd.read_csv('auto-insurance.csv')
    # Calculate number of elements present on the data frame
    elements = len(df)
    x = df.values[:, :-1]
    y = df.values[:, -1]

    # Apply the linear regression by Sklearn library
    regr = linear_model.LinearRegression()
    #Fit linear model
    regr.fit(x, y)

    #Predict using the linear model.
    y_prediction = regr.predict(x)
    print(f"Predicted values:\n {y_prediction}")

    # Predict total payment for all the claims in thousands
    predicted_at_value = regr.predict([[args.prediction]])
    print(f"Prediction for {args.prediction} claims = {predicted_at_value[0]} thousands Swedish Kronors\n")

    # Create a tuple with the number of claims(x) and the predicted
    # payment for the claims (y)
    t_prediction = (args.prediction, predicted_at_value)

    plot_results(x,y, y_prediction, t_prediction)