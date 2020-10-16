from math import log
import sys

def get_nb_accuracy(train_labels, train_docs, test_labels, test_docs, stop_words=[]):
    """
    Train a mulinomial naive Bayes classifer to classify text strings, then
    apply it to each string in a testing set and return the accuracy.

    input: train_labels, train_docs, test_labels, test_docs
    Each of the inputs is a column of a pandas dataframe. train_docs and
    test_docs contain strings of email messages. train_labels and test_labels
    denotes the class/label those messages belong to.

    output: accuracy
    The accuracy of the model trained on train_docs when applied to the test set
    test_docs.
    """
    accuracy = 0.0
    correct = 0
    vocab, class_priors, cond_prob = train_multinomial_nb(train_labels, train_docs, stop_words)
    for i in range(len(test_docs)):
        guess = apply_multinomial_nb(test_labels, vocab, class_priors, cond_prob, test_docs[i])
        if (guess == test_labels[i]):
            correct += 1
    accuracy = correct / len(test_docs)
    return accuracy

def train_multinomial_nb(labels, docs, stop_words=[]):
    """
    Train a multinomial naive Bayes classifier to classify text strings.

    input: labels, docs, stop_words
    labels and docs are columns of a pandas dataframe. docs contains strings of
    email messages. labels denotes the class/label those messages belong to.
    stop_words is a list of words to be ignored.

    output: vocab, class_priors, cond_prob
    vocab is the set of unique words in the entire training text. class_priors
    denotes the prior probability of each class. cond_prob is the conditional
    probability of each word for each class/label.
    """
    labels = list(labels)
    vocab = [word for word in list(set(' '.join(docs).split(' '))) if word not in stop_words]
    num_docs = len(docs)
    unique_labels = set(labels)
    class_priors = {}
    cond_prob = init_cond_prob(vocab, unique_labels)
    for label in unique_labels:
        num_docs_in_class = labels.count(label)
        class_priors[label] = num_docs_in_class / num_docs
        text_in_class = get_text_in_class(docs, label, labels)
        # Remove stop_words
        text_in_class = " ".join([word for word in text_in_class.split(" ") if word not in stop_words])
        for term in vocab:
            tokens_of_term_in_class = text_in_class.count(term)
            cond_prob[term][label] = (tokens_of_term_in_class + 1) / (len(text_in_class) + len(vocab))
    return vocab, class_priors, cond_prob

def apply_multinomial_nb(labels, vocab, class_priors, cond_prob, doc):
    """
    Apply a previously trained multinomial naive Bayes classifier to a doc
    from a test set and get the most likely class/label for that doc.

    input: labels, vocab, class_priors, cond_prob, doc
    labels is a column of pandas dataframe containing the class/labels of each
    string in a test set. vocab is the unique words in training set.
    class_priors is the prior probability of each class/label. cond_prob is the
    conditional probability of each term in each class. doc is a string from a
    test set.

    output: most_likely_class
    The most likely class/label given the words contained in doc.
    """
    labels = list(labels)
    most_likely_class = None
    label_names = list(set(labels))
    # highest_score must be initialized to a very low number, since the sums of
    # log probabilities will be negative numbers.
    highest_score = -9999999
    score = {}
    doc_tokens = [token for token in doc if token in vocab]
    for label in label_names:
        score[label] = log(class_priors[label])
        for token in doc_tokens:
            score[label] += log(cond_prob[token][label])
        if (score[label] > highest_score):
            highest_score = score[label]
            most_likely_class = label
    return most_likely_class

def get_text_in_class(docs, label, labels):
    """
    Helper function to get one large string of all the text matching a given
    label/class.
    """
    text_in_class = ""
    for l in range(len(labels)):
        if (labels[l] == label):
            text_in_class += " " + docs[l]
    return text_in_class

def init_cond_prob(vocab, unique_labels):
    """
    Helper function to initialize the cond_prob dict of conditional probabilities
    for terms in each label/class.
    """
    cond_prob = {}
    for term in vocab:
        cond_prob[term] = {}
        for label in unique_labels:
            cond_prob[term][label] = 0
    return cond_prob

'''

def get_filter_words(docs, max_threshold=0.95):
    """
    Helper function to get words that appear in docs with frequency at or above
    the max_threshold.
    """
    filter_words = []
    word_freqs = {}
    for doc in docs:
        for unique_word in set(doc.split(" ")):
            freq = word_freqs.get(unique_word)
            if (freq):
                word_freqs[unique_word] = freq + 1
            else:
                word_freqs[unique_word] = 1
    for word, freq in word_freqs.items():
        rel_freq = freq / len(docs)
        if (rel_freq >= max_threshold):
            filter_words.append(word)
    return filter_words
'''
