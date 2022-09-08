# Model Analysis

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
