import os
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import math
import numpy as np

"""
Preprocesses input dataset
"""
def data_preprocessing(dataset):
    # Determine whether a column contains numerical or nominial values
    # Create a new Pandas dataframe to maintain order of columns when doing One-Hot Coding on Nominial values
    new_dataframe = pd.DataFrame()
    # Iterate through all the columns of the training_set 
    for x in dataset.columns:
        # Determine if the column 'x' in training set is a Nominial Data or Numerical 
        if is_string_dtype(dataset[x]) and not is_numeric_dtype(dataset[x]):
            # Apply One-Hot Encoding onto Pandas Series at column 'x' 
            dummies = pd.get_dummies(dataset[x], prefix=x, prefix_sep='.', drop_first=True)
            # Combine the One-Hot Encoding Dataframe to our new dataframe to the new_dataframe 
            new_dataframe = pd.concat([new_dataframe, dummies],axis=1)
        else: 
            # If the column 'x' is a Numerical Data, then just add it to the new_dataframe
            new_dataframe[x] = dataset[x]
    return new_dataframe

"""
Function to run an input dataset on Decision Tree Classifier
"""
def run_decision_tree_classifier(X_train, X_test, y_train, y_test):

    # Create a decision tree classifier
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=2,random_state=randomSeed)

    # Train the classifier
    clf = clf.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    # Print the confusion matrix of the classifier
    cm = confusion_matrix(y_test, y_pred)

    # Make confusion matrix pandas dataframe
    cm_df = pd.DataFrame(cm)

    # Calculate the recall of the classifier
    recall = list(recall_score(y_test, y_pred, labels=label_list, average=None))

    return cm_df, accuracy, recall

"""
Function to run an input dataset on Neural Network Classifier
"""
def run_neural_network_classifier(X_train, X_test, y_train, y_test):
    # Create a decision tree classifier
    clf = MLPClassifier(max_iter=1000, random_state=randomSeed)

     # Train the classifier
    clf = clf.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    # Print the confusion matrix of the classifier
    cm = confusion_matrix(y_test, y_pred)

    # Make confusion matrix pandas dataframe
    cm_df = pd.DataFrame(cm)

    # Calculate the recall of the classifier
    recall = list(recall_score(y_test, y_pred, labels=label_list, average=None))

    return cm_df, accuracy, recall

"""
Function to run an input dataset on Random Forest Classifier
"""
def run_random_forest_classifier(X_train, X_test, y_train, y_test):
    # Create a decision tree classifier
    clf = RandomForestClassifier(n_estimators=100, criterion ='entropy', random_state=randomSeed)

     # Train the classifier
    clf = clf.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    # Print the confusion matrix of the classifier
    cm = confusion_matrix(y_test, y_pred)

    # Make confusion matrix pandas dataframe
    cm_df = pd.DataFrame(cm)

    # Calculate the recall of the classifier
    recall = list(recall_score(y_test, y_pred, labels=label_list, average=None))

    return cm_df, accuracy, recall

"""
Function to run an input dataset on naive bayes classifier
"""
def run_naive_bayes_classifier(X_train, X_test, y_train, y_test):
    # Create a decision tree classifier
    clf = MultinomialNB()

     # Train the classifier
    clf = clf.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    # Print the confusion matrix of the classifier
    cm = confusion_matrix(y_test, y_pred)

    # Make confusion matrix pandas dataframe
    cm_df = pd.DataFrame(cm)

    # Calculate the recall of the classifier
    recall = list(recall_score(y_test, y_pred, labels=label_list, average=None))

    return cm_df, accuracy, recall

"""
Calculate confidence interval for a given accuracy
"""
def calculate_confidence_interval(accuracy, test_set_size, number_of_comparisons):
    # Assign Z value using Bonferroni Correction
    if number_of_comparisons < 3: 
        z_value = 1.96
    elif number_of_comparisons == 3:
        z_value = 2.24
    else: 
        z_value = 2.39
    # Calculate the confidence interval
    internal = (z_value * math.sqrt((accuracy * (1-accuracy))/(test_set_size)))
    confidence_interval_array =  [accuracy - internal, accuracy + internal] 
    return confidence_interval_array

"""
Main part of the program
"""
try: 
    # Get random seed from command line
    randomSeed = int(sys.argv[1])

    # Get dataset file path from command line
    dataset = ["monks1.csv"]#,"votes.csv", "hypothyroid.csv", "mnist_1000.csv"]

    # Set size of training set
    training_set_size = 0.6

    # Run decision tree classifier for each dataset in the list
    for input_dataSet in dataset:
        print(f"Current Dataset: {input_dataSet}")
        # Read dataset into a pandas dataframe
        df = pd.read_csv(input_dataSet)
            # Get the target column
        label_col = df.iloc[:,:1]

        # Get the features
        features = df.iloc[:,1:]

        # Preprocess the dataset with one hot encoding
        cleaned_features = data_preprocessing(features)

        # Combine the target column with the features
        df = pd.concat([label_col, cleaned_features], axis=1)

        # Shuffle the dataset and split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(cleaned_features, label_col, test_size=training_set_size, random_state=randomSeed)

        # Flatten the target columns into a 1D array. Prevents a DataConversionWarning warning from being thrown
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # Get the unique labels in the target column
        label_list = np.unique(y_test)

# Decision Tree Classifier
        # Run decision tree classifier
        cm_df, accuracy, recall = run_decision_tree_classifier(X_train, X_test, y_train, y_test)

        # Structure name of file
        filename = f"results/results-DecisionTree-{os.path.splitext(input_dataSet)[0]}-{randomSeed}.csv"

        # Save to file
        cm_df.to_csv(filename)

        # Calculate Confidence Interval
        confidence_interval = calculate_confidence_interval(accuracy, len(y_test), 3)

        # Print results
        print(f"File: {input_dataSet}\nModel: Decision Tree\nConfusion Matrix:\n{cm_df}\nAccuracy Score: {accuracy}\nConfidence Interval: {confidence_interval}")
        count = 0
        for label in label_list:
            print(f"Label '{label}' Recall: {recall[count]}")
            count += 1
        print()

# Neural Network Classifier
        # Run neaural network classifier
        cm_df, accuracy, recall = run_neural_network_classifier(X_train, X_test, y_train, y_test)

        # Structure name of file
        filename = f"results/results-NeauralNetwork-{os.path.splitext(input_dataSet)[0]}-{randomSeed}.csv"

        # Save to file
        cm_df.to_csv(filename)

        # Calculate Confidence Interval
        confidence_interval = calculate_confidence_interval(accuracy, len(y_test), 3)

        # Print results
        print(f"File: {input_dataSet}\nModel: Neural Network\nConfusion Matrix:\n{cm_df}\nAccuracy Score: {accuracy}\nConfidence Interval: {confidence_interval}")
        count = 0
        for label in label_list:
            print(f"Label '{label}' Recall: {recall[count]}")
            count += 1
        print()

# Random Forest Classifier
        # Run random forest classifier
        cm_df, accuracy, recall = run_random_forest_classifier(X_train, X_test, y_train, y_test)

        # Structure name of file
        filename = f"results/results-RandomForest-{os.path.splitext(input_dataSet)[0]}-{randomSeed}.csv"

        # Save to file
        cm_df.to_csv(filename)

        # Calculate Confidence Interval
        confidence_interval = calculate_confidence_interval(accuracy, len(y_test), 3)

        # Print results
        print(f"File: {input_dataSet}\nModel: Random Forest\nConfusion Matrix:\n{cm_df}\nAccuracy Score: {accuracy}\nConfidence Interval: {confidence_interval}")
        count = 0
        for label in label_list:
            print(f"Label '{label}' Recall: {recall[count]}")
            count += 1
        print()

# Naive Bayes Classifier
        # Run naive bayes classifier   
        cm_df, accuracy, recall = run_naive_bayes_classifier(X_train, X_test, y_train, y_test)
        
        # Structure name of file
        filename = f"results/results-NaiveBayes-{os.path.splitext(input_dataSet)[0]}-{randomSeed}.csv"

        # Save to file
        cm_df.to_csv(filename)

        # Calculate Confidence Interval
        confidence_interval = calculate_confidence_interval(accuracy, len(y_test), 3)

        # Print results
        print(f"File: {input_dataSet}\nModel: Naive Bayes\nConfusion Matrix:\n{cm_df}\nAccuracy Score: {accuracy}\nConfidence Interval: {confidence_interval}")
        count = 0
        for label in label_list:
            print(f"Label '{label}' Recall: {recall[count]}")
            count += 1
        print()
  
        print("---" * 20)

except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease correct and try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease correct and try again.")
    exit(1)