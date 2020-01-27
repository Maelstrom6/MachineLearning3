
y_test = [0, 1, 0, 1, 0, 0, 0, 1]
y_pred = [0, 1, 0, 0, 1, 0, 0, 0]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("")
tn = cm[0][0]
fn = cm[1][0]
fp = cm[0][1]
tp = cm[1][1]
accuracy = (cm[0][0] + cm[1][1])/(cm[0][0]+cm[1][0]+cm[0][1]+cm[1][1])
precision = tp/(tp+fp)  # TP/TP+FP
recall = tp/(tp+fn)  # TP/TP+FN
specificity = cm[1][1]/(cm[0][1]+cm[1][1])  # TN/TN+FP
f1 = 2*precision*recall/(precision+recall)
print(precision)
print(recall)
print(f1)
print("")
from sklearn.metrics import f1_score, precision_score, recall_score
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

