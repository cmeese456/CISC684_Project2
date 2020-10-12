import sys
import pandas as pd
import gzip
import os
import glob
import re

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
    combined_df = ham_frame.append(spam_frame).drop_duplicates().reset_index(drop=True)

    #return the combined frame
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

    # remove leading whitespace
    text = re.sub(r"^\s+" , "" , text)

    # return the preprocessed text
    return text

# Testing Functions
test_df = load_all(d1_train)
# print(test_df[0][0])
print(test_df)
