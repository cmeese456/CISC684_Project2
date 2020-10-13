#%%
from numpy import array, dot, random
from random import choice
import pandas as pd
import matplotlib.pyplot as plt

"""
Implement the perceptron algorithm (use the perceptron training rule and not the gradient descent rule).
Notice that unlike logistic regression which is a batch algorithm, the perceptron algorithm is an incremental or stochastic algorithm.
Treat number of iterations in the perceptron algorithm as a hyper-parameter and use the 70-30 split method described earlier to choose 
a suitable value for this hyper-parameter. 
Then, use the chosen value of hyper-parameter, train on the full training dataset and report accuracy on the test set.
"""

# Perceptron training rule

# Do until converge
#   For each x in D
#       For each w_i:
#           update_weights

# The step function classifies any value that is greater than 0 as 1 and everything else as 0
# o(x_1, ..., x_n) = { 1, if w_0 + w_1*x_1 + ... + w_k*x_k > 0
#                    { -1, else
step_function = lambda x: 0 if x < 0 else 1

# x_i: the current value
# t: the target value
# o: the current prediction
# eta: the learning rate, a small value
def calculate_delta_w(current_value, target_value, current_prediction, learning_rate):
    return learning_rate * (target_value - current_prediction) * current_value


def update_weights(w_i, delta_w):
    return w_i + delta_w

# Replace this training set with the real set
training_dataset = [(array([0, 0, 1]), 0), (array([0, 1, 1]), 1), (array([1, 0, 1]), 1), (array([1, 1, 1]), 1)]

# Assign 3 random weights to start
weights = random.rand(3)

# *** Remove this variable later! This is used to catalog all the errors so we can visualize it ***
error = []

# Should we adjust this to be smaller?
learning_rate = 0.2

 # n = number of iterations
n = 100

# Model training
for j in range(n):
    x, expected = choice(training_dataset) # get a random set for input in the training data
    result = dot(weights, x) # calculate the dot product of the input/current value and the weights
    err = expected - step_function(result) # compare the prediction with the expected result
    error.append(err) #*** Remove this line later ***
    # Calculate the change in weights
    # If the expected result is bigger than the actual result, increase the weights
    # If the expected result is smaller than the actual result, decrease the weights
    delta_w = calculate_delta_w(step_function(result), expected, x , learning_rate)
    # Update the weights so we get a better prediction on the next iteration
    weights = update_weights(weights, delta_w)

# Model evaluation
# *** DO WE TEST THIS ON THE VALIDATION SET OR THE TEST SET? ***
for x, _ in training_dataset:
    result = dot(x, weights) # calculate the dot product of the input/current value and the weights
    print('{}: {} -> {}'.format(x[:2], result, step_function(result))) # print the input, the actual value, predicted result, and expected result to see if they match


# For visualization purposes, set the y axis on the graph to be in the range of -1 to 1
plt.ylim([-1, 1])
plt.plot(error) # plot the error to see when the weights converged
plt.show() 

# %%
