import sys
import pandas as pd
import gzip
import os
import glob
import re
import warnings
import MCAP_LR
import multinomial_nb
import dataset_engineering
from perceptron_algorithm import run_perceptron
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Setup path variables for dataset files
d1_train = "dataset_1/train/"
d1_test = "dataset_1/test/"
d2_train = "dataset_2/train/"
d2_test = "dataset_2/test/"
d3_train = "dataset_3/train/"
d3_test = "dataset_3/test/"

# Read in the list of stop words
stop_file = open("stopwords.txt")
stop_list = stop_file.read().splitlines()

# Takes a path to a dataset root folder (so ../dataset/test)
# And builds a pandas dataframe from the data in ham and spam subfolders
# Returns a single dataframe representing all of the training or testing data for a single dataset
#! Note: Input path must end in a trailing "/"
def load_all(path):
    # Create the ham dataframe
    ham_frame = load_data(path, "ham")
    # Create the spam dataframe
    spam_frame = load_data(path, "spam")
    # Combine the frames
    combined_df = ham_frame.append(
        spam_frame).drop_duplicates().reset_index(drop=True)
    # Shuffle the dataframe in-place and reset the index
    # This may not be required but seems like it would be better to randomize the order of the training data
    # Compared to having all hams followed by all spams sequentially.
    combined_df = combined_df.sample(
        frac=1, random_state=123).reset_index(drop=True)
    # return the combined frame
    return combined_df

# Helper function for load_all which handles creating the dataframe for a given subfolder (spam/ham)
def load_data(path, identifier):
    # instantiate empty array and iterator
    i = 0
    df = {}
    # Loop through every file in the path subfolder denoted by identifier (spam or ham)
    # For each file, open it to read, replace new lines with spaces and preprocess the text
    # Then create an object representing each email with (contents, identifier) and add it to the array
    # Lastly convert the array into a dataframe using from_dict and return it
    for x in glob.glob(os.path.join(path+identifier, '*')):
        file_contents = open(x, 'r', errors='ignore').read().replace('\n', ' ')
        file_contents = text_preprocess(file_contents)
        data_item = [file_contents, identifier]
        df[i] = data_item
        i += 1
    big_df = pd.DataFrame.from_dict(df, orient='index')
    return big_df

def load_data_full_path(path, identifier):
     # instantiate empty array and iterator
    i = 0
    df = {}
    # Loop through every file in the path subfolder denoted by identifier (spam or ham)
    # For each file, open it to read, replace new lines with spaces and preprocess the text
    # Then create an object representing each email with (contents, identifier) and add it to the array
    # Lastly convert the array into a dataframe using from_dict and return it
    for x in glob.glob(os.path.join(path, '*')):
        file_contents = open(x, 'r', errors='ignore').read().replace('\n', ' ')
        file_contents = text_preprocess(file_contents)
        data_item = [file_contents, identifier]
        df[i] = data_item
        i += 1
    big_df = pd.DataFrame.from_dict(df, orient='index')
    return big_df

# Function to preprocess the text of an email and prepare it for analysis
def text_preprocess(text):
    # remove all sepcial characters
    text = re.sub(r'\W', ' ', text)
    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # remove all single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # substitute multiple white spaces for a single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # convert to lowercase
    text = text.lower()
    # remove leading whitespace and space characters
    text = text.strip(' ')
    # return the preprocessed text
    return text

# Divide the training set into two sets using a 70/30 split.
#
# Input: a preprocessed (split) training, testing, validation and full training df.
# Returns: processed matrix for each df + a CountVectorizer
# Rows represent emails
# Columns represent words (which are condensed to be numeric) and cells represent frequency counts
def build_features(train_df, test_df, validation_df, full_training_frame):
    # Create the vectorizer for the dataset
    # max_df=0.95 removes "Subject" from the vocabulary since it appears in >95% of emails
    # stop_list is a list of stop words I found online to improve our accuracy.
    # Retrieved from https://github.com/kavgan/nlp-in-practice/blob/master/tf-idf/resources/stopwords.txt
    cv = CountVectorizer(binary=False, max_df=0.95, stop_words=stop_list)
    # Build the dictionary from the full training data
    cv.fit_transform(full_training_frame[0].values)
    # Vectorize each df
    training_matrix = cv.transform(train_df[0].values).toarray()
    testing_matrix = cv.transform(test_df[0].values).toarray()
    validation_matrix = cv.transform(validation_df[0].values).toarray()
    full_training_matrix = cv.transform(full_training_frame[0].values).toarray()
    # Return the four dfs and cv
    return training_matrix, testing_matrix, cv, validation_matrix, full_training_matrix

def run_logistic_regression():
    # Run regression on dataset 1
    print("\nRUNNING LOG REGRESSION ON DATASET 1\n")
    MCAP_LR.driver(d1_train_full_matrix, d1_train_matrix_70, d1_validation_matrix, d1_test_matrix, d1_train_full_labels, d1_test_labels, d1_train_70_labels, d1_validation_labels)
    # Run regression on dataset 2
    print("\nRUNNING LOG REGRESSION ON DATASET 2\n")
    MCAP_LR.driver(d2_train_full_matrix, d2_train_matrix_70, d2_validation_matrix, d2_test_matrix, d2_train_full_labels, d2_test_labels, d2_train_70_labels, d2_validation_labels)
    # Run regression on dataset 3
    print("\nRUNNING LOG REGRESSION ON DATASET 3\n")
    MCAP_LR.driver(d3_train_full_matrix, d3_train_matrix_70, d3_validation_matrix, d3_test_matrix, d3_train_full_labels, d3_test_labels, d3_train_70_labels, d3_validation_labels)

# Preprocess the stop list so it is consistent with our input
for x in stop_list:
    x = text_preprocess(x)

# Load all of the data files
d1_train_full = load_all(d1_train)
d1_test = load_all(d1_test)
d2_train_full = load_all(d2_train)
d2_test = load_all(d2_test)
d3_train_full = load_all(d3_train)
d3_test = load_all(d3_test)

# Split each training dataset into a 70/30 group with 70% training and 30% validation
# Suppress FutureWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    d1_train_70, d1_validation = train_test_split(d1_train_full, random_state = 456, train_size = .7)
    d2_train_70, d2_validation = train_test_split(d2_train_full, random_state = 456, train_size = .7)
    d3_train_70, d3_validation = train_test_split(d3_train_full, random_state = 456, train_size = .7)

# Generate a labels array for each matrix representing the label for each row in the matrix
# Since a row represents an email, an example is as follows:
# d1_train_70[0] represents an array containing the frequency counts for all words in the first email of the set
# and d1_train_70_labels[0] contains the label for the email held in d1_train_70[0] (either "ham" or "spam")
# D1 labels
d1_train_full_labels = d1_train_full[1].values
d1_test_labels = d1_test[1].values
d1_train_70_labels = d1_train_70[1].values
d1_validation_labels = d1_validation[1].values
# D2 labels
d2_train_full_labels = d2_train_full[1].values
d2_test_labels = d2_test[1].values
d2_train_70_labels = d2_train_70[1].values
d2_validation_labels = d2_validation[1].values
# D3 labels
d3_train_full_labels = d3_train_full[1].values
d3_test_labels = d3_test[1].values
d3_train_70_labels = d3_train_70[1].values
d3_validation_labels = d3_validation[1].values

# Create the Matrix representation for each dataset and its associated frames
#! Use these matrices + the labels arrays as input to your model
d1_train_matrix_70, d1_test_matrix, d1_cv, d1_validation_matrix, d1_train_full_matrix = build_features(d1_train_70, d1_test, d1_validation, d1_train_full)
d2_train_matrix_70, d2_test_matrix, d2_cv, d2_validation_matrix, d2_train_full_matrix = build_features(d2_train_70, d2_test, d2_validation, d2_train_full)
d3_train_matrix_70, d3_test_matrix, d3_cv, d3_validation_matrix, d3_train_full_matrix = build_features(d3_train_70, d3_test, d3_validation, d3_train_full)

def run_the_models(model_to_run="all"):
    if model_to_run == "all":
        # Run the NB model
        print("Running Naive Bayes model...")
        nb_accuracy_d1 = multinomial_nb.get_nb_accuracy(d1_train_full[1], d1_train_full[0], d1_test[1], d1_test[0])
        print("Naive Bayes accuracy on data set 1: " + str(nb_accuracy_d1))
        nb_accuracy_d2 = multinomial_nb.get_nb_accuracy(d2_train_full[1], d2_train_full[0], d2_test[1], d2_test[0])
        print("Naive Bayes accuracy on data set 2: " + str(nb_accuracy_d2))
        nb_accuracy_d3 = multinomial_nb.get_nb_accuracy(d3_train_full[1], d3_train_full[0], d3_test[1], d3_test[0])
        print("Naive Bayes accuracy on data set 3: " + str(nb_accuracy_d3))

        # Run the MCAP Model
        run_logistic_regression()

        # Run the Perceptron Model
        run_perceptron('dataset_1/train/ham', 'dataset_1/test/ham', 'dataset_1/train/spam', 'dataset_1/test/spam')
        run_perceptron('dataset_2/train/ham', 'dataset_2/test/ham', 'dataset_2/train/spam', 'dataset_2/test/spam')
        run_perceptron('dataset_3/train/ham', 'dataset_3/test/ham', 'dataset_3/train/spam', 'dataset_3/test/spam')


    if model_to_run == 'NB' or model_to_run == 'nb' or model_to_run == 'Nb':
        # Run the NB model
        print("Running Naive Bayes model...")
        nb_accuracy_d1 = multinomial_nb.get_nb_accuracy(d1_train_full[1], d1_train_full[0], d1_test[1], d1_test[0])
        print("Naive Bayes accuracy on data set 1: " + str(nb_accuracy_d1))
        nb_accuracy_d2 = multinomial_nb.get_nb_accuracy(d2_train_full[1], d2_train_full[0], d2_test[1], d2_test[0])
        print("Naive Bayes accuracy on data set 2: " + str(nb_accuracy_d2))
        nb_accuracy_d3 = multinomial_nb.get_nb_accuracy(d3_train_full[1], d3_train_full[0], d3_test[1], d3_test[0])
        print("Naive Bayes accuracy on data set 3: " + str(nb_accuracy_d3))

    if model_to_run == 'MCAP' or model_to_run == 'mcap' or model_to_run == 'Mcap':
        run_logistic_regression()

    if model_to_run == 'Perceptron' or model_to_run == 'PERCEPTRON' or model_to_run == 'perceptron':
        # Run the Perceptron Model
        print('~~RUNNING PERCEPTRON ON DATASET 1~~')
        run_perceptron('dataset_1/train/ham', 'dataset_1/test/ham', 'dataset_1/train/spam', 'dataset_1/test/spam')
        print('~~RUNNING PERCEPTRON ON DATASET 2~~')
        run_perceptron('dataset_2/train/ham', 'dataset_2/test/ham', 'dataset_2/train/spam', 'dataset_2/test/spam')
        print('~~RUNNING PERCEPTRON ON DATASET 3~~')
        run_perceptron('dataset_3/train/ham', 'dataset_3/test/ham', 'dataset_3/train/spam', 'dataset_3/test/spam')


# Driver to run the program with command line arg
if len(sys.argv) == 2:
    run_the_models(sys.argv[1])
else:
    run_the_models()
