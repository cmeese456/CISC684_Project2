import sys
import pandas as pd
import math
from math import exp
import numpy as np

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

def new_make_prediction(instance, parameters):
    # Define a prediction variable and add w0 to it
    prediction = parameters[0]

    # For a given Xi, compute Wi * Xi. Sum the results across all Xi
    weighted_sum = 0
    for i in range (len(instance) - 1):
        weighted_sum += instance[i] * parameters[i+1]

    # Add the weighted sum to w0 to form our complete prediction
    prediction += weighted_sum

    # Return the prediction value for classification
    try:
        return 1.0 / (1.0 + exp(-prediction))
    except OverflowError as err:
        return 1
        print("Encountered overflow error during prediction")


# Gradient Ascent function to learn the parameters.
# n_rounds is the number of training cycles
def gradient_ascent(training_set, classifications, learn_rate, n_rounds, lambda_value):
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
            predicted_y = new_make_prediction(instance, parameters)

            # Update the error
            error = classifications[i] - predicted_y
            total_error += error**2

            # Update W0 coefficient
            #parameters[0] = parameters[0] + learn_rate * 1 * (classifications[i] - predicted_y) - (learn_rate * lambda_value * parameters[0])
            #parameters[0] = parameters[0] + learn_rate * 1 * (classifications[i] - predicted_y)
            parameters[0] = parameters[0] + learn_rate * 1 * error * predicted_y * (1.0 - predicted_y) - (learn_rate * lambda_value * parameters[0])

            # Third Loop to update all coefficients Wi
            for j in range(len(instance)-1):
                #parameters[j+1] = parameters[j+1] + learn_rate * instance[j] * (classifications[i] - predicted_y) - (learn_rate * lambda_value * parameters[j+1])
                #parameters[j+1] = parameters[j+1] + learn_rate * instance[j] * (classifications[i] - predicted_y)
                parameters[j+1] = parameters[j+1] + learn_rate * instance[j] * error * predicted_y * (1.0 - predicted_y) - (learn_rate * lambda_value * parameters[j+1])
            # Increment i
            i += 1

        # Print some results of the training round
        print('>~traing_round=%d, learning_rate=%.3f, error=%.3f' % (round, learn_rate, total_error))

    # Return the parameters array
    return parameters

# Runs the model on a testing set and outputs the accuracy
def test_the_model(testing_set, classifications, weights):
    # Create variables for the number of correct classifications and accuracy
    num_correct = 0
    accuracy = 0.0

    # Transform the classifications from textual to numeric
    for z in range(len(classifications)):
        if (classifications[z] == 'ham'):
            classifications[z] = 1
        if (classifications[z] == 'spam'):
            classifications[z] = 0

    # Loop through the testing data and do tests
    i = 0
    for row in testing_set:
        prediction = 0
        predicted_y = make_prediction(row, weights)
        if(predicted_y >= .5):
            prediction = 1
        if(predicted_y < .5):
            prediction = 0

        # Compare our prediction to the true value
        if(prediction == classifications[i]):
            num_correct += 1

        # Increment i
        i += 1

    accuracy = num_correct / len(testing_set)
    return accuracy


# Function to drive the entire MCAP_LR Procedure
def driver(full_training_set, training_set_70, validation_set, testing_set, train_class, testing_class, train_70_class, validation_class):
    # Define necessary inputs to train the model
    learning_rate = 0.4
    num_training_rounds = 50
    num_training_rounds_lambda = 25
    lambda_value = 0.0

    print("LEARNING RATE=%.2f, TRAINING_ROUNDS=%d, 70/30_VALIDATION_ROUNDS=%d" % (learning_rate, num_training_rounds, num_training_rounds_lambda))

    # Learn the parameters using the 70% training set
    # We will define a list of suitable parameters, test them on the 70% training set
    # And select the best result to use for full training
    potential_lambda = [.01, .005, .004, .003, .002, .001, .0005, .0001]
    lambda_accuracy = 0.0

    # Test each potential lambda value
    for x in potential_lambda:
        print("~~~~Testing lambda value %.4f" % x)
        parameters_70 = gradient_ascent(training_set_70, train_70_class, learning_rate, num_training_rounds_lambda, x)
        new_accuracy = test_the_model(validation_set, validation_class, parameters_70)
        print("DONE TRAINING WITH Lambda Value=%.4f, Validation_Accuracy=%.3f" % (x, new_accuracy))
        if (new_accuracy > lambda_accuracy):
            lambda_accuracy = new_accuracy
            lambda_value = x

    # Output the chosen lambda value
    print("Chosen_Lambda=%.4f, Best_Validation_Accuracy=%.3f" % (lambda_value, lambda_accuracy))

    # Use this lambda value to learn from the entire training set
    learned_parameters = gradient_ascent(full_training_set, train_class, learning_rate, num_training_rounds, lambda_value)

    # Test the new parameters on the testing set
    accuracy = test_the_model(testing_set, testing_class, learned_parameters)

    # Report the final results
    print("The final accuracy on the testing set is: %.3f" % accuracy)
