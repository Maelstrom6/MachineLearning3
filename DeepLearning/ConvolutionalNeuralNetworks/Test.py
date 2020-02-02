from keras import backend as K
gpus = K.tensorflow_backend._get_available_gpus()

print(gpus)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

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


# import the data set
dataset = pd.read_csv("Social_Network_Ads.csv")
dataset = dataset.drop(["User ID"], axis=1)
dataset = encode_one_hot(dataset)
print(dataset)

# make sure X is a matrix not a vector
x = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3].values

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# PCA
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=2, kernel="rbf")  # linear, rbf, poly and sigmoid kernels
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Fitting logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from matplotlib.colors import ListedColormap
#X_set, y_set = X_test, y_test
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=[ListedColormap(("red", "green", "blue"))(i)], label=j)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = [ListedColormap(('red', 'green'))(i)], label = j)

plt.legend()
plt.show()