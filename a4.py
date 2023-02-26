

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv('winequality-red.csv', sep=";")


"""
The prepData function is being used to create the test sets and training sets of data
The variables that are returned are the variables that will be used for training and testing the machine
input_train, output_train, input_test, output_test are all converted to arrays that can be used in the machine
Uncomment print statements to see the returned values/outputs of the prepData function
"""


def prepData(data):
    train_len = int(len(data) * 0.8)
    test_len = len(data) - train_len - 1
    input_train = np.array(data.iloc[0:train_len, 0:11])
    output_train = np.array(data.iloc[0:train_len, 11:12])
    input_test = np.array(data.iloc[train_len:train_len + test_len, 0:11])
    output_test = np.array(data.iloc[train_len:train_len + test_len, 11:12])
    return input_train, output_train, input_test, output_test
# print(prepData(data))


"""
a = array for input_train
b = array for output_train
c = array for input_test
d = array for output_test
Uncomment print statements to see the arrays a, b, c, and d
"""
a, b, c, d = prepData(df)
# print(d)
"""
The dTree function makes a decision tree with the output of the decision tree classifier being plotted as a confusion 
matrix. The accuracy is also included in the function. In other words, the function also prints how accurate the machine
model was with respect to the output test values. Uncomment print statements to print out the predicted labels.  The 
accuracy score (defined as score) is used to determine the accuracy of the decision tree in % to 2 d.p. 
"""


def dTree(v1, v2, v3, v4):
    clf = tree.DecisionTreeClassifier()
    clf.fit(v1, v2)
    clf.predict(v3)
    # print(clf.predict(v3))
    pred = clf.predict(v3)
    plot_confusion_matrix(clf, v3, v4, normalize='all', cmap=plt.cm.Purples)
    plt.title('Confusion matrix for Decision Tree')
    score = accuracy_score(v4, pred) * 100
    print("%.2f" % score)
    return plt.show()


"""
The supportVM function makes a support virtual machine with the output of the support virtual machine being plotted as a 
confusion matrix. The accuracy is also included in the function. In other words, the function also prints how accurate
the machine model was with respect to the output test values. Uncomment print statements to print out the predicted
labels. The accuracy score (defined as score) is used to determine the accuracy of the SVM in % to 2 d.p.
Normalize or not?
"""


def supportVM(v1, v2, v3, v4):
    cls = svm.SVC(kernel='linear')
    cls.fit(v1, v2.ravel())
    cls.predict(v3)
    # print(cls.predict(v3))
    pred = cls.predict(v3)
    score = accuracy_score(v4, pred) * 100
    plot_confusion_matrix(cls, v3, v4.ravel(), normalize='all', cmap=plt.cm.Blues)
    plt.title('Confusion matrix for Support Vector Machine')
    print("%.2f" % score)
    return plt.show()


# CALL FUNCTIONS DOWN BELOW
prepData(df)
dTree(a, b, c, d)
supportVM(a, b, c, d)
