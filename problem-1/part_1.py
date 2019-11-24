import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creating the decision tree
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

# print the confusion matrix
cfmx = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=['True : 1', 'True : 0'],
    columns=['Pred : 1', 'Pred : 0']
)
print('Confusion Matrix:\n{}'.format(cfmx))

# print the report data
report = classification_report(y_test, y_pred, output_dict=True)
print('\nAccuracy:         {:0.3f}'.format(report['accuracy']))
print('Precision for 1:  {:0.3f}'.format(report['1']['precision']))
print('Recall for 1:     {:0.3f}'.format(report['1']['recall']))
print('Precision for 0:  {:0.3f}'.format(report['0']['precision']))
print('Recall for 0:     {:0.3f}'.format(report['0']['recall']))

# create a png image of the decision tree as `part-1.png`
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=['x', 'y'],
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('part-1.png')
Image(graph.create_png())

# test all the points with the decision tree
all_point_pred = classifier.predict(X)
dataset.insert(3, 'Predicted', all_point_pred, True)

# graph for the predicted chart
color = ['red' if c == 1 else 'black' for c in dataset.Predicted]
plt.scatter(dataset.X, dataset.Y, c=color)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('predicted_clusters.png', bbox_inches='tight')
plt.show()

# graph for the true chart
color = ['red' if c == 1 else 'black' for c in dataset.Class]
plt.scatter(dataset.X, dataset.Y, c=color)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('true_clusters.png', bbox_inches='tight')
plt.show()
