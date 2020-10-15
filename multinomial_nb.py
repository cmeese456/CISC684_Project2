from math import log
from operator import add
import sys

# Practical version
def train_multinomial_nb(labels, train_matrix):
    labels = list(labels)
    label_names = list(set(labels))
    # Unique words/terms (i.e. the columns in the dataframe)
    labeled_tokens = get_labeled_tokens(labels, train_matrix) # Not the same thing as vocab; will have to ajdust things
    # Number of total documents (i.e. number of rows in the dataframe)
    num_docs = len(train_matrix)
    # Priors for each class
    class_priors = {}
    # Conditional probabilities for words in each class
    cond_prob = {}
    vocab_size = get_vocab_size(train_matrix)
    for label in label_names:
        cond_prob[label] = {}
        num_docs_in_class = labels.count(label)
        class_priors[label] = num_docs_in_class / num_docs
        num_tokens_in_class = len([token for token in labeled_tokens[label] if token != 0])
        for term in range(len(train_matrix[0])):
            tokens_of_term_in_class = labeled_tokens[label][term]
            cond_prob[label][term] = (tokens_of_term_in_class + 1) / (num_tokens_in_class + vocab_size)
    return labeled_tokens, class_priors, cond_prob

def get_labeled_tokens(labels, train_matrix):
    # We want lists for the frequencies of each word for each label
    labeled_tokens = {}
    for label in list(set(labels)):
        # Initialize lists of each word's frequency to 0
        labeled_tokens[label] = [0] * len(train_matrix[0])
    # Iterate through labels and rows of documents to sum each word's frequency
    # for each label. Keep those sums lists stored in the labeled_tokens dict.
    for i in range(len(labels)):
        for key in labeled_tokens.keys():
            if (labels[i] == key):
                labeled_tokens[key] = list(map(add, train_matrix[i], labeled_tokens[key]))
    return labeled_tokens

def get_vocab_size(train_matrix):
    # train_matrix could contain columns that equal 0 in every row. We want to
    # ignore those columns to find the size of the vocab that's actually
    # present in the training set.
    vocab = [0] * len(train_matrix[0])
    for i in range(len(train_matrix)):
        vocab = list(map(add, train_matrix[i], vocab))
    vocab_size = len([term for term in vocab if term != 0])
    return vocab_size

# Practical version
def apply_multinomial_nb(labels, class_priors, cond_prob, doc):
    labels = list(labels)
    most_likely_class = None
    label_names = list(set(labels))
    highest_score = -9999999
    score = {}
    for label in label_names:
        score[label] = log(class_priors[label])
        # I.e. for each frequency in doc, which is a row from a test matrix
        for token in range(len(doc)):
            score[label] += log(cond_prob[label][token])
        if (score[label] > highest_score):
            highest_score = score[label]
            most_likely_class = label
    return most_likely_class

def get_nb_accuracy(train_labels, test_labels, train_matrix, test_matrix):
    labeled_tokens, class_priors, cond_prob = train_multinomial_nb(train_labels, train_matrix)
    num_correct = 0
    accuracy = 0.0
    for i in range(len(test_matrix)):
        chosen_class = apply_multinomial_nb(test_labels, class_priors, cond_prob, test_matrix[i])
        if (chosen_class == test_labels[i]):
            num_correct += 1
    accuracy = num_correct / len(test_matrix)
    return accuracy

'''
# Version from pseudocode
def train_multinomial_nb(labels, docs):
    classes = list(set(labels))
    # Unique words/terms (i.e. the columns in the dataframe)
    vocab = extract_vocabulary(docs)
    # Number of total documents (i.e. number of rows in the dataframe)
    num_docs = count_docs(docs)
    # Priors for each class
    class_priors = {}
    # Conditional probabilities for words in each class
    cond_prob = init_cond_prob(classes, docs)
    for class in classes:
        num_docs_in_class = count_docs_in_class(docs, class)
        class_priors[class] = num_docs_in_class / num_docs
        text_in_class = concatenate_text_of_all_docs_in_class(docs, class)
        for term in vocab:
            tokens_of_term_in_class = count_tokens_of_term(text_in_class, term)
            cond_prob[term][class] = (tokens_of_term_in_class + 1) / (len(text_in_class) + len(vocab))
    return vocab, class_priors, cond_prob

# Version from pseudocode
def apply_multinomial_nb(labels, vocab, class_priors, cond_prob, doc):
    labels = list(labels)
    most_likely_class = None
    label_names = list(set(labels))
    highest_score = 0
    score = {}
    doc_tokens = extract_tokens_from_doc(vocab, doc)
    for label in label_names:
        score[label] = log(class_priors[label])
        for token in doc_tokens:
            score[label] += log(cond_prob[token][label])
        if (score[label] > highest_score):
            highest_score = score[label]
            most_likely_class = label
    return most_likely_class
'''
