
import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.formulatools

def find_abs_min(arr):
    """
    Finds minimum value of the array where each element is absoluted
    :param arr: the input array
    :return: minimum value and its first index
    """
    min_val, min_index = (abs(arr[0]), 0)
    for i in range(len(arr)):
        if min_val > abs(arr[i]):
            min_val, min_index = (abs(arr[i]), i)
    return min_val, min_index

dataset = pd.read_csv("50_Startups.csv")

X = dataset.values[:, : -1]
# X = dataset.iloc[:, : -1].values
y = dataset.values[:, 4]

ct_X = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],
    remainder='passthrough')
X = np.array(ct_X.fit_transform(X), dtype=np.float)

# Avoiding the dummy variable trap
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the results
y_pred = regressor.predict(X=X_test)

# building the optimal model with backwards elimination
# setup variables for loop
X = np.append(arr=np.ones((len(X), 1)).astype(np.float), values=X, axis=1)
columns = [0, 1, 2, 3, 4, 5]  # Used colims
critical_value = 0.1
running = True


# This is working
while running:
    # Setup the new smaller set with more significant variables
    running = False
    X_opt = X[:, columns]
    X_opt = X_opt.astype(float)
    regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
    print(type(regressor_ols))
    print(regressor_ols.tvalues)

    # Get t values to find the lowest one and remove it from the set
    t_values = regressor_ols.tvalues
    val, index = find_abs_min(t_values)
    if val < critical_value:
        running = True
        columns.remove(index)
print(columns)
