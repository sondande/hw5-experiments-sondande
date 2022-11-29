import os
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


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
            # Find the maximum value in column 'x'
            max_value = max(dataset[x])
            # Find the minimum value in column 'x'
            min_value = min(dataset[x])
            # Check if the column being evaluated is the label column. If so, just add it right into the dataframe
            if x =='label':
                new_dataframe = pd.concat([new_dataframe, dataset[x]], axis=1)
                continue
            # Ensure we don't run into a zero division error when normalizing all the values
            elif (max_value - min_value) != 0:
                # Apply net value formula to every value in pandas dataframe
                dataset[x] = dataset[x].apply(lambda y: (y - min_value)/(max_value - min_value))
                # Combine New column to our new_dataframe
                new_dataframe = pd.concat([new_dataframe, dataset[x]],axis=1)
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
    return cm_df, accuracy

"""
Function to run an input dataset on Neural Network Classifier
"""

def run_neural_network_classifier(X_train, X_test, y_train, y_test):
    # Create a decision tree classifier
    clf = MLPClassifier(max_iter=5000, random_state=randomSeed)

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
    return cm_df, accuracy

"""
Main part of the program
"""

try: 
    # Get random seed from command line
    randomSeed = int(sys.argv[1])

    # Get dataset file path from command line
    dataset = ["monks1.csv"]#,"votes.csv"]

    # Set size of training set
    training_set_size = 0.6


    # Run decision tree classifier for each dataset in the list
    for input_dataSet in dataset:
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

    # Run decision tree classifier
    cm_df, accuracy = run_decision_tree_classifier(X_train, X_test, y_train, y_test)

    # Structure name of file
    filename = f"results-DecisionTree-{os.path.splitext(input_dataSet)[0]}-{randomSeed}.csv"

    # Save to file
    cm_df.to_csv(filename)

    print(f"File: {input_dataSet}\nModel: Decision Tree\nConfusion Matrix:\n{cm_df}\nAccuracy Score: {accuracy}\n")

    # Run neaural network classifier
    cm_df, accuracy = run_neural_network_classifier(X_train, X_test, y_train, y_test)

    # Structure name of file
    filename = f"results-NeauralNetwork-{os.path.splitext(input_dataSet)[0]}-{randomSeed}.csv"

    # Save to file
    cm_df.to_csv(filename)

    print(f"File: {input_dataSet}\nModel: Neural Network\nConfusion Matrix:\n{cm_df}\nAccuracy Score: {accuracy}\n")



except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease correct and try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease correct and try again.")
    exit(1)