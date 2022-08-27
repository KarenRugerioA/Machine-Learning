# Author: Karen Rugerio
# Implementation of the linear regression without using machine learning existing libraries

# Dataset used: 
# https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/
# Dependant variables -> No. of Cylinders (2nd column) 
# Independant variable -> Hoursepower (4th column)

from fileinput import filename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "./auto-mpg.data"

# Only for debugging
pd.set_option('display.max_columns', None) 

def mean(read_data)->float:
    """
    Function to calculate the mean value
    """
    sum_total = sum(read_data)
    elements = len(read_data)
    mean = sum_total / float(elements)

    return mean


def variance(read_data)->float:
    """
    Function to calculate the variance
    """
    mean_value = mean(read_data)
    mean_difference_squared = [pow((reading - mean_value), 2) for reading in read_data]
    variance = sum(mean_difference_squared)

    return variance / float(len(read_data) - 1)


def covariance(read_var1, read_var2)->float:
    """
    Function to calculate the covariance
    """
    covariance_result = 0.0
    mean_var1 = mean(read_var1)
    mean_var2 = mean(read_var2)
    readings_size = len(read_var1)

    for i in range(0, readings_size):
        covariance_result += (read_var1[i] - mean_var1) * (read_var2[i] - mean_var2)

    return covariance_result / float(readings_size - 1)


def get_coefficients(x_value, y_value):
    """
    Function to calculate linear regression coefficients
    """
    b1 = covariance(x_value, y_value) / float(variance(x_value))
    b0 = mean(y_value) - (b1 * mean(x_value))
    
    return b0, b1

def get_mse(actual_value, predicted_value)->float:

    error_total = 0.0
    elements = len(actual_value)
    for i in range(0, elements):
        difference = actual_value[i] - predicted_value[i]
        squared_difference = pow(difference, 2)
        error_total += squared_difference
    
    mse = error_total / elements
    
    return mse

def predict_horsepower(x_value, w0, w1)->None:
    df["PredictedHorsepower"] = w0 + w1 * x_value
    print(df[["horsepower", "PredictedHorsepower"]])

def linear_regression(df):
    """
    Linear Regression implementation
    """
    cylinders_mean = mean(df["cylinders"].astype(float))
    print("Cylinders mean: %f" % cylinders_mean)
    horsepower_mean = mean(df["horsepower"].astype(float))
    print("Horsepower mean: %f" % horsepower_mean)

    cylinders_variance = variance(df["cylinders"].astype(float))
    print("Cylinders variance: %f" % cylinders_variance)
    horsepower_variance = variance(df["horsepower"].astype(float))
    print("Horsepower variance: %f" % horsepower_variance)

    covariance_cylinders_horsepower = covariance(df["cylinders"].astype(float), df["horsepower"].astype(float))
    print("Covariance of Cylinders and horsepower: %f" % covariance_cylinders_horsepower)
    w0 , w1 = get_coefficients(df["cylinders"].astype(float), df["horsepower"].astype(float))
    print("Coefficient w0 %f" % w0)
    print("Coefficient w1 %f" %w1)

    """
    Predict horse power
    """
    predict_horsepower(df["cylinders"].astype(float), w0, w1)
    mse = get_mse(df["horsepower"], df["PredictedHorsepower"])
    #print("Mean Square Error: %f" % mse)

    print (np.polyfit(df["cylinders"].astype(float), df["horsepower"].astype(float), deg=2))
    plt.plot(df["cylinders"].astype(float),df["horsepower"].astype(float), 'yo', df["PredictedHorsepower"], '--k' )

if __name__ == '__main__':

    columns = ["mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model year",
    "origin",
    "car name"]

    df = pd.read_csv(filename, sep='\s+',header=None, names=columns)
    linear_regression(df)
    
