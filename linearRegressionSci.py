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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

    #Defining dependant and independant values
    x = df.values[:, :-1] #Inependant variable (number of claims)
    y = df.values[:, -1] #Dependant variable (cost in thousands Swedish Kronor)

    """Separating test and train data, in this case, the 
    test sample is going to be 20% of the dataset and the
    train sample is going to be the 80% of the dataset
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.20, random_state=1)

    # Apply the linear regression by Sklearn library
    regr = linear_model.LinearRegression()

    #Fit linear model
    regr.fit(x_train, y_train)

    #Predict using the linear model.
    y_prediction = regr.predict(x_train)
    print(f"Predicted values:\n {y_prediction}")

    # Accuracy calculation of the train based on MSE
    score = r2_score(y_train, y_prediction)
    print(f"The score (based on train MSE) is: {score}\n")

    # Predict salary in USD for years of experience (INPUT)
    predicted_at_value = regr.predict([[args.prediction]])
    print(f"Prediction for {args.prediction} ensurance payment = {predicted_at_value[0]} Swedish Kronor\n")
    
    # Predict salary based on years of experience with test values
    y_predTest = regr.predict(x_test)

    # Accuracy calculation of the test based on MSE
    score = r2_score(y_test, y_predTest)
    print(f"The score (based on test MSE) is: {score}\n")

    # Create a tuple with the years of experience(x) and the predicted salary (y)
    t_prediction = (args.prediction, predicted_at_value)

    # Create a new array to include the x of the input given by the user and y predicted value
    new_x = np.append(x_test, t_prediction[0])
    new_y_prediction = np.append(y_predTest, t_prediction[1])

    plt.scatter(x_train, y_train, color='green', alpha=0.4)
    plt.scatter(x_test, y_predTest, color = 'black', alpha=0.7)
    plt.scatter(t_prediction[0], t_prediction[1], color='#050df7')
    plt.annotate('Predicted value', xy=t_prediction, horizontalalignment='right', verticalalignment='top')
    plt.title("xtest-ypredtest")
    plt.plot(new_x, new_y_prediction, color='#F5451F')
    plt.legend(['TRAIN SET', 'TEST SET','PREDICTED VALUE', 'FITTED FUNCTION'])
    plt.show()

    #plot_results(Xtrain, ytrain, y_prediction, t_prediction)