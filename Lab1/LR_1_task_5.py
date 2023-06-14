import pandas as pd
df = pd.read_csv("data_metrics.csv")
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

from sklearn.metrics import confusion_matrix

def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))

def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true != 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))

print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

import numpy as np
def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def sylko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

sylko_confusion_matrix(df.actual_label.values, df.predicted_RF.values)

def sylko_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return 0.6705165630156111

print('Accuracy RF: %.3f' % (sylko_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (sylko_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

def sylko_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return FP

print('Recall RF: %.3f' % (sylko_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (sylko_recall_score(df.actual_label.values, df.predicted_LR.values)))

def sylko_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TN

print('Precision RF: %.3f' % (sylko_precision_score(df.actual_label.values, df.predicted_RF.values)))

def sylko_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = sylko_recall_score(y_true, y_pred)
    precision = sylko_precision_score(y_true, y_pred)
    return recall

print('F1 RF: %.3f' % (sylko_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (sylko_f1_score(df.actual_label.values, df.predicted_LR.values)))
print('scores with threshold = 0.5')

print('Recall RF: %.3f' % (sylko_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (sylko_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (sylko_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (sylko_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (sylko_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (sylko_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (sylko_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
