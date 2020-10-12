import sys
import pandas as pd
import math

# Prediction function for learning and evaluation
# Instance represents a single instance of the dataset which includes email body text and a classification (0,1)
#! parameters[0] should always be w0
def make_prediction(instance, parameters):
    # Define a prediction variable and add w0 to it
    prediction = parameters[0]

    # For a given Xi, compute Wi * Xi. Sum the results across all Xi
    weighted_sum = 0
    for i in range (len(instance) - 1):
        weighted_sum += instance[i] * parameters[i+1]

    # Add the weighted sum to w0 to form our complete prediction
    prediction += weighted_sum

    # Return the prediction value for classification
    return prediction

# Divide the training set into two sets using a 70/30 split.
def splt_data(dataframe):
    # Split the data into two new frames, one 70% one 30%

    # Return both frames in a list
    return ""

# Learn parameters using the 70% split

# Use the 30% of data as a validation set to select lambda

# Then use selected value of Lambda to learn parameters from the full training set

# Use gradient ascent for learning the weights (set appropriate learning rate)
# Implement a hard limit on the number of iterations for gradient ascent

# Implement the algorithm
