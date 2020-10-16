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
# learning_rate represents eta in the equation for calculating the change in weight
# In this function we run the perceptron using the perceptron learning rule until convergence occurs
# the iteration when convergence happened is returned
def learn_weights_by_convergence(data_set, weights, learning_rate):
    misclassified = True
    iteration = 0
    while misclassified:
        num_misclassified = 0
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
                if calculate_delta_w(current_value[weight], target_value, current_prediction, learning_rate) > 0:
                    num_misclassified += 1

        # the perceptron converges when data is classified correctly
        # if the number of points that are misclassified = 0, we reached convergence
        if num_misclassified == 0:
            iteration += 1
            misclassified = False
        else:
            iteration += 1

    # return the iteration number where the perceptron converged
    return iteration


# data_set is the set to learn the weights for (typically the training set)
# weights is the list of weights
# n is the number of iterations to run the perceptron training rule
# learning_rate represents eta in the equation for calculating the change in weight
# In this function we run the perceptron using the perceptron learning rule until a certain number of iterations
# The number of iterations is determined by a previous run of the perceptron when convergence occurs
def learn_weights_by_iteration(data_set, weights, n, learning_rate):
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


# Classify either 0 or 1
def perceptron_classifier(test_set, weights):
    weight_sum = weights['w_0']
    key_list = list(test_set.keys())
    for k in key_list:
        if k not in weights:
            weights[k] = 0.0
        weight_sum += weights[k] * test_set[k]
        classification = step_function(weight_sum)

    return classification


# Find how accurate the perceptron is by taking the number of predictions we got correct
# divided by the entire length of the test set
def get_accuracy(correct_predictions, test_set):
    print("Correctly Guessed: " + str(int(correct_predictions)))
    print("Total Guessed: " + str(len(test_set)))
    print('Perceptron Accuracy:', str(
        (correct_predictions/len(test_set)) * 100) + str('%'))


# Run the perceptron using the number of iterations we learned by finding convergence
def run_perceptron_using_learned_iterations(ham_train_set_path, ham_test_set_path, spam_train_set_path, spam_test_set_path, n):
    learning_rate = 0.01  # initialize the learning rate to a small number
    n = int(n)  # convert the number of iterations from a string to an int

    # create the data sets
    train_set_ham = make_dataset(ham_train_set_path, 0)
    train_set_spam = make_dataset(spam_train_set_path, 1)

    test_set_ham = make_dataset(ham_test_set_path, 0)
    test_set_spam = make_dataset(spam_test_set_path, 1)

    total_train_set = train_set_ham + train_set_spam
    total_test_set = test_set_ham + test_set_spam

    # initialize the weight of each word as 0
    bag_of_words = get_unique_words(total_train_set)

    weights = {'w_0': 1.0}

    for word in bag_of_words:
        weights[word] = 0.0

    # learn each weight
    learn_weights_by_iteration(total_train_set, weights, n, learning_rate)

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

    print('***ACCURACY OF PERCEPTRON ON TEST SET TRAINED ON ALL TRAINING DATA')
    get_accuracy(correct_predictions, total_test_set)
    print()


# Run the perceptron using the number of iterations we learned by finding convergence on the
# validation set
def run_perceptron_using_learned_iterations_on_validation_set(validation_set, n):
    learning_rate = 0.01  # initialize the learning rate to a small number
    n = int(n)  # convert the number of iterations from a string to an int

    # initialize the weight of each word as 0
    bag_of_words = get_unique_words(validation_set)

    weights = {'w_0': 1.0}

    for word in bag_of_words:
        weights[word] = 0.0

    # learn each weight
    convergence_iteration = learn_weights_by_convergence(
        validation_set, weights, learning_rate)

    correct_predictions = 0
    for i in range(len(validation_set)):
        current_value = validation_set[i][2]
        prediction = perceptron_classifier(current_value, weights)
        if prediction == 1:
            if validation_set[i][1] == 1:
                correct_predictions += 1
        elif prediction == 0:
            if validation_set[i][1] == 0:
                correct_predictions += 1

    print('***ACCURACY OF PERCEPTRON ON VALIDATION SET LEARNING ITERATIONS')
    get_accuracy(correct_predictions, validation_set)
    print('NUMBER OF ITERATIONS FOR CONVERGENCE:',
          str(convergence_iteration), '\n')

    print('RUNNING PERCEPTRON ON TEST SET USING THE NUMBER OF ITERATIONS FOR CONVERGENCE THAT THE VALIDATION SET FOUND...\n')
    run_perceptron_using_learned_iterations('dataset_1/train/ham/', 'dataset_1/test/ham/',
                                            'dataset_1/train/spam/', 'dataset_1/test/spam/', convergence_iteration)


# run the perceptron and the associated perceptron run functions
def run_perceptron(ham_train_set_path, ham_test_set_path, spam_train_set_path, spam_test_set_path):

    learning_rate = 0.01  # initialize the learning rate to a small number

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

    total_test_set = test_set_ham + test_set_spam

    # split the total training set into 70% training data and 30% validation data

    # initialize the weight of each word as 0
    bag_of_words = get_unique_words(train_set_70)

    weights = {'w_0': 1.0}

    for word in bag_of_words:
        weights[word] = 0.0

    # learn each weight
    convergence_iteration = learn_weights_by_convergence(
        train_set_70, weights, learning_rate)

    correct_predictions_test = 0
    for i in range(len(total_test_set)):
        current_value = total_test_set[i][2]
        prediction = perceptron_classifier(current_value, weights)
        if prediction == 1:
            if total_test_set[i][1] == 1:
                correct_predictions_test += 1
        elif prediction == 0:
            if total_test_set[i][1] == 0:
                correct_predictions_test += 1

    correct_predictions_validation = 0
    for i in range(len(validation_set)):
        current_value = total_test_set[i][2]
        prediction = perceptron_classifier(current_value, weights)
        if prediction == 1:
            if validation_set[i][1] == 1:
                correct_predictions_validation += 1
        elif prediction == 0:
            if validation_set[i][1] == 0:
                correct_predictions_validation += 1

    print('***ACCURACY OF PERCEPTRON ON TEST SET TRAINED ON 70% OF TRAINING DATA:')
    get_accuracy(correct_predictions_test, total_test_set)
    print('NUMBER OF ITERATIONS FOR CONVERGENCE:',
          str(convergence_iteration), '\n')

    print('***ACCURACY OF PERCEPTRON ON VALIDATION SET USING LEARNED ITERATIONS:')
    get_accuracy(correct_predictions_validation, validation_set)
    print()

    run_perceptron_using_learned_iterations('dataset_1/train/ham/', 'dataset_1/test/ham/',
                                            'dataset_1/train/spam/', 'dataset_1/test/spam/', convergence_iteration)

    run_perceptron_using_learned_iterations_on_validation_set(
        validation_set, convergence_iteration)
