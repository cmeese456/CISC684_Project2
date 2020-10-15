import sys
import pandas as pd
import math
from math import exp

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
    # If prediction >= 0 then Spam (0)
    # If Prediction < 0 we assign Ham (1)
    return prediction

# Gradient Ascent function to learn the parameters.
# n_rounds is the number of training epochs
#! TODO: Implement L2 Regularization
def gradient_ascent(training_set, classifications, learn_rate, n_rounds):
    # Create an empty parameters list
    parameters = [0.0 for i in range(len(training_set[0]))]

    # Transform the classifications from textual to numeric
    for z in range(len(classifications)):
        if (classifications[z] == 'ham'):
            classifications[z] = 1
        if (classifications[z] == 'spam'):
            classifications[z] = 0

    # First loop which determines number of training rounds
    for round in range(n_rounds):
        # Define the total error
        total_error = 0

        # Define a loop control variable for grabbing the classification
        i = 0

        # Second Loop for each row in the training set
        for instance in training_set:
            # Make a prediction using the current coefficients
            predicted_y = make_prediction(instance, parameters)

            # Update the error
            error = classifications[i] - predicted_y
            total_error += error**2

            # Update W0 coefficient
            parameters[0] = parameters[0] + learn_rate * 1 * (classifications[i] - (exp(parameters[0] + parameters[0]*1)/(1+exp(parameters[0] + parameters[0]*1))))

            # Third Loop to update all coefficients Wi
            for j in range(len(instance)-1):
                parameters[j+1] = parameters[j+1] + learn_rate * instance[j] * (classifications[i] - (exp(parameters[0] + parameters[j+1]*instance[j]) / (1+(exp(parameters[0] + parameters[j+1]*instance[j])))))

            # Increment i
            i += 1
        # Print some results of the training round
        print('>traing round=%d, learning rate=%.3f, error=%.3f' % (round, learn_rate, total_error))

    # Return the parameters array
    return parameters


# Learn parameters using the 70% split

# Use the 30% of data as a validation set to select lambda

# Then use selected value of Lambda to learn parameters from the full training set

# Use gradient ascent for learning the weights (set appropriate learning rate)
# Implement a hard limit on the number of iterations for gradient ascent

# Implement the algorithm
