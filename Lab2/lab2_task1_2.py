import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
X_encoded = np.empty(X.shape)
label_encoder = [] 
for i, item in enumerate(X[0]):
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifierPoly = SVC(kernel='poly', degree=4)
classifierPoly.fit(X_train, y_train)
y_test_pred = classifierPoly.predict(X_test)

num_folds = 3
accuracy_values = cross_val_score(classifierPoly, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifierPoly, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifierPoly, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

f1 = cross_val_score(classifierPoly, X, y, scoring='f1_weighted', cv=num_folds)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifierRbf = SVC(kernel='rbf')
classifierRbf.fit(X_train, y_train)
y_test_pred = classifierRbf.predict(X_test)

num_folds = 3
accuracy_values = cross_val_score(classifierRbf, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifierRbf, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifierRbf, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

f1 = cross_val_score(classifierRbf, X, y, scoring='f1_weighted', cv=num_folds)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifierSigmoid = SVC(kernel='sigmoid')
classifierSigmoid.fit(X_train, y_train)
y_test_pred = classifierSigmoid.predict(X_test)

num_folds = 3
accuracy_values = cross_val_score(classifierSigmoid, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifierSigmoid, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifierSigmoid, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

f1 = cross_val_score(classifierSigmoid, X, y, scoring='f1_weighted', cv=num_folds)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
