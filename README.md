# CISC684_Project2
Second project for CISC684

To run all models, enter the following at the command line

```
python3 project2_executable.py
```

To run one model at a time, enter:

```
python3 project2_executable.py modelname
```

where ```modelname``` is one of the following: ```nb```, ```mcap```, or ```perceptron```.

In this project, we implement and evaluate Naive Bayes, Perceptron, and Logistic Regression for text classification.

Specifically, we implement a multinomial Naive Bayes algorithm for text classification as described here: https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf. We do all calculations in log-scale to to avoid overflow. Next, we implement the MCAP Logistic Regression algorithm with L2 regularization. We use gradient ascent to learn the weights, but put a hard limit on the number of iterations gradient ascent runs for to save time. Finally, we implement the perceptron algorithm using the perceptron training rule.
