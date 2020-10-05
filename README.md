# CISC684_Project2
Second project for CISC684

In this project, we implement and evaluate Naive Bayes, Perceptron, and Logistic Regression for text classification.

Specifically, we implement a multinomial Naive Bayes algorithm for text classification as described here. We do all calculations in log-scale to to avoid overflow. Next, we implement the MCAP Logistic Regression algorithm with L2 regularization. We use gradient ascent to learn the weights, but put a hard limit on the number of iterations gradient ascent runs for to save time. Finally, we implement the perceptron algorithm using the perceptron training rule.