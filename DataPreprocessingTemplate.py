import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# import the data set
dataset = pd.read_csv("Data.csv")

# First part of the bracket describes the rows of the dataset
# The colon means that we want all the lines of the dataset
# The right of the comma are the columns of the dataset
# We want all the columns except the last column
# .values converts to python syntax
x = dataset.iloc[:, : -1].values

# Create the dependent variable vector
# The index of the last column
y = dataset.iloc[:, 3].values

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


def remove_english_old(data):
    # This does not avoid the dummy variable trap
    data = pd.DataFrame(data)
    print(data.dtypes)
    columns = []
    for i, item in enumerate(data.iloc[0, :].values):
        if type(item) == str:
            columns.append(i)
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(sparse=False), columns)],  # range(data.iloc[0, :].values)
        remainder='passthrough')
    data = np.array(ct.fit_transform(data), dtype=np.float)
    return pd.DataFrame(data)

# Encode categorical data without warnings
# The column numbers to be transformed (here is [0] but can be [0, 1, 3])
# Leave the rest of the columns untouched
ct_x = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
print(x)
print(type(x))
x = np.array(ct_x.fit_transform(x), dtype=np.float)
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
print(x)
print(type(x))

# Splitting dataset into training set and test set
# Test size is usually 0.25
# Usually don't use random_state
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


print(x)
print(y)
