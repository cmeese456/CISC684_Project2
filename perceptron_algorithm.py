"""
Implement the perceptron algorithm (use the perceptron training rule and not the gradient descent rule).
Notice that unlike logistic regression which is a batch algorithm, the perceptron algorithm is an incremental or stochastic algorithm.
Treat number of iterations in the perceptron algorithm as a hyper-parameter and use the 70-30 split method described earlier to choose 
a suitable value for this hyper-parameter. 
Then, use the chosen value of hyper-parameter, train on the full training dataset and report accuracy on the test set.
"""

# Perceptron training rule

# x_i: the current value
# t: the target value
# o: the current prediction
# eta: the learning rate, a small value
def calculate_delta_w(current_value, target_value, current_prediction, learning_rate):
    return learning_rate * (target_value - current_prediction) * current_value


def update_weights(w_i, delta_w):
    return w_i + delta_w


# o(x_1, ..., x_n) = { 1, if w_0 + w_1*x_1 + ... + w_k*x_k > 0
#                    { -1, else
    
# Do until converge
#   For each x in D
#       For each w_i:
#           update_weights


