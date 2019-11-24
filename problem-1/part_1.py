import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# creating visual
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'



# import the dataset from the Excel file
dataset = pd.read_excel("../P4Data.xlsx")

# remove the classifier from the data
X = dataset.drop('Class', axis=1)  # data to be tested
y = dataset['Class']               # the classifier column

# use train_test_split with a test size of 30%, train size of 70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# create a decision tree classifier and give it the train and test data
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# test the decision tree with the test data
y_pred = classifier.predict(X_test)

# print the confusion matrix and other metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=['x', 'y'],
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('part-1.png')
Image(graph.create_png())
