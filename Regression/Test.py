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
print("hello there"[-1])
