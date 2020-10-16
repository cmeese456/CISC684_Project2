# %%
# import pandas as pd
# from project2_executable import load_all
from dataset_engineering import make_dataset, get_word_frequency, split_train_set_to_70_train_30_validation, get_unique_words

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

# The step function classifies any value that is greater than 0 as 1 (SPAM) and everything else as 0 (HAM)
# o(x_1, ..., x_n) = { 1, if w_0 + w_1*x_1 + ... + w_k*x_k > 0
#                    { -1, else


def step_function(x): return 0 if x < 0 else 1


# x_i: the current value
# t: the target value
# o: the current prediction
# eta: the learning rate, a small value
def calculate_delta_w(current_value, target_value, current_prediction, learning_rate):
    return learning_rate * (target_value - current_prediction) * current_value


# data_set is the set to learn the weights for (typically the training set)
# weights is the list of weights
# n is the number of iterations to run the perceptron training rule
# learning_rate represents eta in the equation for calculating the change in weight
def learn_weights(data_set, weights, n, learning_rate):
    i = 0
    while i < n:
        # For every file in the dataset
        for x in range(len(data_set)):
            weight_sum = weights['w_0']
            current_value = data_set[x][2]

            # Update the weights for every file found in the dataset
            for w in current_value:
                if w not in weights:
                    weights[w] = 0.0
                weight_sum += weights[w] * current_value[w]
            current_prediction = 0.0
            if weight_sum > 0:
                current_prediction = 1.0
            target_value = 0.0

            # If the target value in our dataset is 1, update that to be 1
            if data_set[x][1] == 1:
                target_value = 1.0
            for weight in current_value:
                weights[weight] += calculate_delta_w(
                    current_value[weight], target_value, current_prediction, learning_rate)
        i += 1


def perceptron_classifier(test_set, weights):
    weight_sum = weights['w_0']
    key_list = list(test_set.keys())
    for k in key_list:
        if k not in weights:
            weights[k] = 0.0
        weight_sum += weights[k] * test_set[k]
        classification = step_function(weight_sum)

    return classification


# %%
def run_perceptron(ham_train_set_path, ham_test_set_path, spam_train_set_path, spam_test_set_path, n):

    learning_rate = 0.01  # initialize the learning rate to a small number
    n = int(n)  # convert the number of iterations from a string to an int

    # create the data sets

    train_set_ham = make_dataset(ham_train_set_path, 0)
    train_set_spam = make_dataset(spam_train_set_path, 1)

    train_set_70_ham, validation_set_ham = split_train_set_to_70_train_30_validation(
        train_set_ham)
    train_set_70_spam, validation_set_spam = split_train_set_to_70_train_30_validation(
        train_set_spam)
    train_set_70 = train_set_70_ham + train_set_70_spam

    validation_set = validation_set_ham + validation_set_spam

    test_set_ham = make_dataset(ham_test_set_path, 0)
    test_set_spam = make_dataset(spam_test_set_path, 1)

    total_train_set = train_set_ham + train_set_spam
    total_test_set = test_set_ham + test_set_spam

    # split the total training set into 70% training data and 30% validation data

    # initialize the weight of each word as 0
    bag_of_words = get_unique_words(train_set_70)

    weights = {'w_0': 1.0}

    for word in bag_of_words:
        weights[word] = 0.0

    # learn each weight
    learn_weights(train_set_70, weights, n, learning_rate)

    correct_predictions = 0
    for i in range(len(total_test_set)):
        current_value = total_test_set[i][2]
        prediction = perceptron_classifier(current_value, weights)
        if prediction == 1:
            if total_test_set[i][1] == 1:
                correct_predictions += 1
        elif prediction == 0:
            if total_test_set[i][1] == 0:
                correct_predictions += 1

    get_accuracy(correct_predictions, total_test_set)


def get_accuracy(correct_predictions, test_set):
    print("Correctly Guessed: " + str(int(correct_predictions)))
    print("Total Guessed: " + str(len(test_set)))
    print('Perceptron Accuracy:', str(
        (correct_predictions/len(test_set)) * 100) + str('%'))


run_perceptron('dataset_1/train/ham/', 'dataset_1/test/ham/',
               'dataset_1/train/spam/', 'dataset_1/test/spam/', 100)
