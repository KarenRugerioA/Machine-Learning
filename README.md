# Model Analysis

## Dependencies 

```console
pip3 install pandas
pip3 install -U scikit-learn
pip3 install mlxtend
```

## Bias-Variance tradeoff

High bias is a sign of the model missing important relations between features and target outputs.
This is related to using a simple model instead of a more complex one.
- High training error
- Test error is extremely similar as training error

Higher variance is an error from sensitivity to fluctuationsin the training set.
- Low Training error
- High test error

Bias and Variance tradeoff:
1. High Bias and Low Variance = Underfitting. Predictions are mostly consistent but innacurate on average.
                             It happens when the model is too simple with very few parameters. 
2. High Bias and High Variance = Predictions are inconsistent and innacurate on average
3. Low Bias and Low Variance = The ideal model. Difficult to achieve.
4. Low Bias and High Variance = Overfitting. Predictions are mostly inconsistent but accurate on average.
                             This happens when the model is too complex with a large number of parameters. 

## Running the model

```console
user@system:~% python3 modelAnalysis.py
Y predictions: [ 49.49886882  67.58483121  70.59915828  43.47021469 396.14648132
 145.95733491 182.12925969  61.55617708 206.24387621 145.95733491
 133.90002665  61.55617708  82.65646654]
MSE: 1671.7795956092089
Bias: 1384.4849490557654
Variance: 287.2946465534371
```

## Analysis

In this scenario, the Bias is way larger than the Variance: This is an indicator of an **underfitting** model.
Moreover, both, the Bias and the Variance are high. Which could lead to some inconsistent and some innacurate
predictions. To improve the results it would be useful to have more data and/or more parameters.
