
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import keras
# import tensorflow
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump, load
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def encode_one_hot(data):
    """

    :param data: The pandas DataFrame to perform the one hot encoding
    :return: The original data but all text fields are replaced by one hot values
    """
    data = pd.DataFrame(data)
    columns = []  # a list of column names
    new_columns = pd.DataFrame()

    for i, item in enumerate(data.iloc[0, :].values):
        if type(item) == str:
            columns.append(data.columns[i])
            new_column = pd.get_dummies(data.iloc[:, i], drop_first=True, dtype=np.float)
            new_columns = pd.concat([new_columns.reset_index(drop=True), new_column.reset_index(drop=True)], axis=1)

    data = data.drop(columns, axis=1)
    data = pd.concat([new_columns.reset_index(drop=True), data.reset_index(drop=True)], axis=1)
    return pd.DataFrame(data)


def encode_labels(data):
    """

    :param data: The pandas DataFrame to perform the label encoding
    :return: The original dataFrame but text fields are replaced by their integer values
    It also returns an array of columns with 1 if the column is a category and 0 otherwise
    """
    column_categories = []
    data = pd.DataFrame(data)
    print(data.dtypes)
    for i, item in enumerate(data.iloc[0, :].values):
        if type(item) == str:
            column_categories.append(1)
            data[data.columns[i]] = LabelEncoder().fit_transform(data[data.columns[i]])
        else:
            column_categories.append(0)
    return data, column_categories


dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# le_X = LabelEncoder()
# X[:, 0] = le_X.fit_transform(X[:, 0])
# ohe = OneHotEncoder(categories=[0])
# X = ohe.fit_transform(X).toarray()
# le_y = LabelEncoder()
# y=le_y.fit_transform(y)
X = encode_one_hot(X)
X = X.iloc[:, :].values
#y = encode_labels(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting XGBoost
# perfoance, execuation speed, no feature scaling needed
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X)

print(f1_score(y, y_pred))
cm = confusion_matrix(y, y_pred)
print(cm)


# Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y_train, cv=10)  # n_jobs = -1 for threading
avg = accuracies.mean()
var = accuracies.std()
# avg = sum(accuracies)/len(accuracies)
# var = (sum([a*a for a in accuracies]) - avg**2)/len(accuracies)
print(avg)
print(var)

