# This file is used for engineering the dataset
import os
import re
from collections import Counter

# Get the number of times each word appears in a text file
def get_word_frequency(some_text):
    word_frequency = Counter(re.findall(r'\w+', some_text))
    return dict(word_frequency)

# Turn the series of text files into a dataset
def make_dataset(directory, classification):
    dataset = []
    for file_entry in os.listdir(directory):
        file_path = os.path.join(directory, file_entry)
        is_file = os.path.isfile(file_path)
        if is_file:
            with open(file_path, 'r', encoding='Latin-1') as txt_file:
                text = txt_file.read()
                dataset.append([text.split(), classification, get_word_frequency(text)])
    return dataset

# Split the training set into 70% of its original size and
# make a validation set with the remaining data
def split_train_set_to_70_train_30_validation(training_set):
    training_set_70 = []
    validation_set = []
    for i in range(len(training_set)):
        if i <= (len(training_set) * 0.7):
            training_set_70.append(training_set[i])
        else:
            validation_set.append(training_set[i])

    return training_set_70, validation_set

# get a list of all the unique words in a text file to assemble a bag of words
def get_unique_words(txt_files):
    bag_of_words = []
    for text in txt_files:
        for word in text[0]:
            bag_of_words.append(word)

    return list(set(bag_of_words))
