# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

from InputPreparation import get_audiofilenames
import FeatureExtraction


def get_feature_file(path):
    """
    Method to get the file with audio features for classification
    path = path to directory of the file
    """
    df = pd.read_csv(path)
    X = df.values.tolist()  # X = features
    print(" x feature matrix")
    print("---------------------------")
    for row in X:
        print(row)
    return X


# Test the above function
# get_feature_file("Data/test_fake_features_for_DT.csv")

def get_targets_table(path):
    audio_file_names = get_audiofilenames(path)
    first_letter = ""
    y = list()
    i = 0
    for audiofile in audio_file_names:
        first_letter = audiofile[0]
        # print(first_letter)
        if first_letter == "N":
            y.append("N")
        else:
            y.append("Y")
        i += 1
    print(" y target values ")
    print("---------------------------")
    print(y)
    return y


# Test the above function
# get_targets_table("Data/test_DT/")

def draw_dt(clf, feature_cols):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['N', 'Y'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('DT.png')
    Image(graph.create_png())

def generate_col_names():
    col_names = []
    for i in range(0, 12):
        col_names.append("MFCC_coef_"+str(i))
    col_names.append("E")
    for i in range(0, 12):
        col_names.append("MFCC_coef_D_"+str(i))

    for i in range(0, 12):
        col_names.append("MFCC_coef_DD_"+str(i))
    col_names.append("E_D")
    col_names.append("E_DD")
    col_names.append("Pitch")
    col_names.append("Pitch_E")

    return col_names


def apply_desicion_tree(path_to_files, feature_file):
    print("start the apply_desicion_tree")
    col_names = generate_col_names()
    print(col_names)
    print("past generate_col_names")

    feature_cols = col_names
    # load dataset
    # feature_matrix = pd.read_csv(feature_file, header=None, names=col_names, skiprows=1)


    # Use FeatureExtraction class's method to extract the MFCC and pitch features, load them to a dataframe and give that to feature_matrix
    feature_matrix = FeatureExtraction.extract_features(path_to_files)
    print("past load feature_matrix")
    feature_matrix.head()
    numcols = len(feature_matrix[0])
    print("test feature matrix number of columns  ")
    print("----------------")
    print(numcols)

    X = feature_matrix[feature_cols]  # Features
    print("test X ")
    print("----------------")
    print(X)
    #y = pima.label  # Target variable
    y = get_targets_table(path_to_files)
    print("test y ")
    print("----------------")
    print(y)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    print("X_train ")
    print("----------------")
    print(X_train)
    print("X_test ")
    print("----------------")
    print(X_test)
    print("y_train ")
    print("----------------")
    print(y_train)
    print("y_test ")
    print("----------------")
    print(y_test)
    print("----------------")

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # with some optimisation parameters
    # criterion="entropy"
    # criterion = "gini" <-- default
    # max_depth=2
    # max_depth=None <-- default
    #clf = DecisionTreeClassifier(criterion="entropy", max_depth=None)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("y_pred")
    print("----------------")
    print(y_pred)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Report : ")
    print("----------------")
    print(metrics.classification_report(y_test, y_pred))
    draw_dt(clf, feature_cols)
    print(" tree drawn, named test_DT2.png")


apply_desicion_tree("Data/test_DT2/", "Data/mfcc_features.csv")

