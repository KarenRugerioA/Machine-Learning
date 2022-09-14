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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.model_selection import learning_curve

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
    print(f"Predicted values:\n {y_prediction}\n")

    # Accuracy calculation of predictions based on training set using r2
    score = r2_score(y_train, y_prediction)
    print(f"The r2 using train set is: {score}\n")

    # Predict payment in Swedish Kronor for a given number of claims (INPUT)
    predicted_at_value = regr.predict([[args.prediction]])
    
    # Predict payment in Swedish Kronor based on test values (number of claims)
    y_predTest = regr.predict(x_test)

    # Accuracy calculation of predictions based on test set using r2
    score = r2_score(y_test, y_predTest)
    print(f"The r2 using test set is: {score}\n")

    # Cross Validation
    score_CV = cross_val_score(regr, x, y, cv=2)
    print("Accuracy of %0.2f with a standard deviation of %0.2f" % (score_CV.mean(), score_CV.std()))

    # Create a tuple with the number of claims(x) and the Swedish kronor to pay (y)
    t_prediction = (args.prediction, predicted_at_value)

    # Create a new array to include the x of the input given by the user and y predicted value
    new_x = np.append(x_test, t_prediction[0])
    new_y_prediction = np.append(y_predTest, t_prediction[1])

    # A learning curve shows the validation and training score of an estimator for varying numbers of training samples
    train_sizes, train_scores, valid_scores = learning_curve(regr, x, y, cv=3)

    # Calculating train's mean and train's standard deviation obtained when calculating the learning curve
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculating test's means and test's standard deviation obtained when calculating the learning curve
    test_mean = np.mean(valid_scores, axis=1)
    test_std = np.std(valid_scores, axis=1)

    # Applied Lasso regression for standardadization purposes
    lasso_model = linear_model.Lasso()
    lasso_model.fit(x_train, y_train)
    pred_yR = lasso_model.predict(x_test)
    score_CV = cross_val_score(regr, x, y, cv=3)
    print("Accuracy of %0.2f after applying Lasso with a standard deviation of %0.2f" % (score_CV.mean(), score_CV.std()))
    print(f"Prediction for {args.prediction} claims = {predicted_at_value[0]} Swedish Kronor\n")

    # Plotting the learning curve: Training score vs Cross-validation score.
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(train_sizes, train_mean, 'o-', color="red",  label="Training score")
    plt.plot(train_sizes, test_mean,"o-",color="green", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="red")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2,color="green")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score")
    plt.legend(loc="best")

    # Plotting the Linear Regression
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.scatter(x_train, y_train, color='green', alpha=0.4)
    plt.scatter(x_test, y_predTest, color = 'black', alpha=0.7)
    plt.scatter(t_prediction[0], t_prediction[1], color='#050df7')
    plt.annotate('Predicted value', xy=t_prediction, horizontalalignment='right', verticalalignment='top')
    plt.title("Linear Regression")
    plt.plot(new_x, new_y_prediction, color='#F5451F')
    plt.legend(['TRAIN SET', 'TEST SET','PREDICTED VALUE', 'FITTED FUNCTION'])
    plt.show()