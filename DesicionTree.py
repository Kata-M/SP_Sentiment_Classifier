# Importing the required packages

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from InputPreparation import get_audiofilenames
import graphviz as g

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np


def get_feature_file(path):
    """
    Method to get the file with audio features for classification
    path = path to directory of the file

    """
    df = pd.read_csv(path)
    X = df.values.tolist()  # X = features
    for row in X:
        print(row)

    return X


# Test the above function
# get_feature_file("Data/test_fake_features_for_DT.csv")

def get_targets_table(path):
    audio_file_names = get_audiofilenames(path)
    # make a two dimensional array with [ filename, stressed, non-stressed] eg [ N_p4_1_q1.wav , 0 , 1 ]
    first_letter = ""
    y = list()
    i = 0
    for audiofile in audio_file_names:
        first_letter = audiofile[0]
        print(first_letter)
        if first_letter == "N":
            y.append("N")
        else:
            y.append("Y")
        i +=1


    for yrow in y:
        print(yrow)

    return y


# Test the above function
# get_targets_table("Data/test_DT/")

def split_to_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    print(" split to test and train done! ")
    return X_train, X_test, y_train, y_test


# Test the above function
# split_to_train_test(X ,y )

def create_dt_instance(x_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)


def main():
    X = get_feature_file("Data/test_fake_features_for_DT.csv")
    y = get_targets_table("Data/test_DT/")

    X_train, X_test, y_train, y_test = split_to_train_test(X, y)

    print("------- x train ------------")
    print(X_train)
    print("--------x test -----------")
    print(X_test)
    print("-------- y train -----------")
    print(y_train)
    print("--------- y test ----------")
    print(y_test)

    #create_dt_instance(X_train, y_train)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # decision tree produced by our model

    """
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data, feature_names=["a1" , "a2" , "a3" , "a4"])
    (graph,) = graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    """

    # see how the model performs on the test data
    y_pred = dt.predict(X_test)
    i = 0
    for y in y_pred:
        print(y)
        """ 
        if int(y[2]) == y_test[i][2]:
            print(" -- correct -- ")
            print(y[0])
            print(y[1])
            print(y[2])
            print(y_test[i][0])
            print(y_test[i][1])
            print(y_test[i][2])
        elif int(y[2]) != y_test[i][2]:
            print(" --- not correct --- ")
            print(y[0])
            print(y[1])
            print(y[2])
            print(y_test[i][0])
            print(y_test[i][1])
            print(y_test[i][2])
        """
        i += 1

    print("y pred")
    print(y_pred)
    #species = np.array(y_test).argmax(axis=1)
    print("species")
   # print(species)
   # predictions = np.array(y_pred).argmax(axis=1)
    print("predictions")
  #  print(predictions)
   # cf = confusion_matrix(species, predictions)
    print("confusion matrix")
    #print(cf)
    print("main DONE")


# Test the above function
main()

# from this on is from source: https://www.geeksforgeeks.org/decision-tree-implementation-python/
#it is for inspiration; delete later
# Function to split the dataset
def splitdataset(balance_data):
    # Separating the target variable

    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


# Driver code
def main2():
    # Building Phase
    #data = importdata()
    #X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    X = get_feature_file("Data/test_fake_features_for_DT.csv")
    Y = get_targets_table("Data/test_DT/")

    X_train, X_test, y_train, y_test = split_to_train_test(X, Y)

    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

    data = export_graphviz(clf_entropy, feature_names=["a1", "a2", "a3", "a4"])
    graph = g.Source(data, format="png")
    graph.render("./results/ds_{}".format(1))

# Calling main function
 #if __name__ == "__main__":
main2()
