# Author Karen Rugerio
# Model Analysis:
# Calculate the bias and variance for a machine learning model to estimate the error of a model and divide the error down into bias and variance components.
# This will help us to understand how well the model is performing.

"""
Dataset:
Swedish Committee on Analysis of Risk Premium in Motor Insurance
http://college.hmco.com/mathematics/brase/understandable_statistics/7e/students/datasets/
       slr/frames/frame.htmlownlee/Datasets/blob/master/auto-insurance.csv
x = number of claims.
y = total payment for all the claims in thousands.
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp

if __name__ == '__main__':
    
    # Read the dataset
    df = read_csv('auto-insurance.csv')
    data = df.values
    x, y = data[:, :-1], data[:, -1]

    # Split data [80% of train data and 20% of test data (test_size=0.2)]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Define model
    regr = LinearRegression()
    
    # Calculate bias and variance decompression, using loss as 'mse' and 5000 rounds
    mse, bias, variance_r = bias_variance_decomp(regr, x_train, y_train, x_test, y_test, loss='mse', num_rounds=5000, random_seed=1)

    y_pred=regr.predict(x_test)
    print(f"Y predictions: {y_pred}")

    # Results
    print(f"MSE: {mse}")
    print(f"Bias: {bias}")
    print(f"Variance: {variance_r}")


    """
    High bias is a sign of the model missing important relations between features and target outputs.
    This is related to using a simple model instead of a more complex one.
        - High training error
        - Test error is extremely similar as training error

    Higher variance is an error from sensitivity to fluctuationsin the training set.
        - Low Training error
        - High test error

    Bias and Variance trade-off:

    High Bias and Low Variance = Underfitting. Predictions are mostly consistent but innacurate on average.
                                It happens when the model is too simple with very few parameters. 
    High Bias and High Variance = Predictions are inconsistent and innacurate on average
    Low Bias and Low Variance = The ideal model. Difficult to achieve.
    Low Bias and High Variance = Overfitting. Predictions are mostly inconsistent but accurate on average.
                                This happens when the model is too complex with a large number of parameters.
    """