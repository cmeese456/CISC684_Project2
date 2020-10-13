import sys
import pandas as pd
import gzip
import os
import glob
import re
import MCAP_LR
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Command line compiling stuff here:

# Setup path variables for dataset files
d1_train = "dataset_1/train/"
d1_test = "dataset_1/test/"
d2_train = "dataset_2/train/"
d2_test = "dataset_2/test/"
d3_train = "dataset_3/train/"
d3_test = "dataset_3/test/"

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


def splt_data(dataframe):
    # Split the data into two new frames, one 70% one 30%

    # Return both frames in a list
    return ""

# Input: a preprocessed (split) training, testing, validation and full training df.
# Returns: processed matrix for each df + a CountVectorizer
# Rows represent emails
# Columns represent words (which are condensed to be numeric) and cells represent frequency counts
def build_features(train_df, test_df, validation_df, full_training_frame):
    # Create the vectorizer for the dataset
    cv = CountVectorizer(binary=False, max_df=0.95)
    # Build the dictionary from the full training data
    cv.fit_transform(full_training_frame[0].values)
    # Vectorize each df
    training_matrix = cv.transform(train_df[0].values).toarray()
    testing_matrix = cv.transform(test_df[0].values).toarray()
    validation_matrix = cv.transform(validation_df[0].values).toarray()
    full_training_matrix = cv.transform(full_training_frame[0].values).toarray()
    # Return the four dfs and cv
    return training_matrix, testing_matrix, cv, validation_matrix, full_training_matrix


# Testing Functions
test_df = load_all(d1_test)
test_labels = test_df[1].values
train_df = load_all(d1_train)
training_matrix, testing_matrix, cv, validation_matrix, full_training_matrix = build_features(test_df, train_df, test_df, train_df)
print(training_matrix.shape)
print(testing_matrix.shape)
print(validation_matrix.shape)
print(full_training_matrix.shape)
#cv = CountVectorizer(binary=False, max_df=0.95)
#cv = cv.fit_transform(test_df[0].values).toarray()
# print(cv.vocabulary_)
# print(cv.shape)
# print(cv[0][0])
#word_count = 0
# for x in range (0, 10375):
#    word_count += cv[0][x]
#print("The sum is: " , word_count)
# print(test_df[0][0])
# print(test_labels.size)
# build_features(1,2)
# build_features(1,2,3)

#test_df['text'] = test_df[0]
# print(test_df['text'])
# print(test_df[0][404])
# print(test_df)
