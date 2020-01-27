import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

"""
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 0:1].values
ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0])],  # range(data.iloc[0, :].values)
        remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)
print(x)
print(type(x))
"""

'''
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, : -1].values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

ct_x = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
x = np.array(ct_x.fit_transform(x), dtype=np.float)
'''

# import the data set
#dataset = pd.read_csv("Data.csv")
dataset = pd.read_csv("Position_Salaries.csv")

# First part of the bracket describes the rows of the dataset
# The colon means that we want all the lines of the dataset
# The right of the comma are the columns of the dataset
# We want all the columns except the last column
# .values converts to python syntax
x = dataset.iloc[:, : -1].values

# Create the dependent variable vector
# The index of the last column
# y = dataset.iloc[:, 3].values

# Replaces missing data with the average
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode categorical data from tutorial with warnings
'''
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])
one_hot_encoder_x = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder_x.fit_transform(x).toarray()
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
'''

# Encode categorical data without warnings
# The column numbers to be transformed (here is [0] but can be [0, 1, 3])
# Leave the rest of the columns untouched
ct_x = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(sparse=False), [0])],
    remainder='passthrough'
)
print(x)
print(type(x))
x = np.array(ct_x.fit_transform(x), dtype=np.float)
label_encoder_y = LabelEncoder()
print(x)
print(type(x))


