import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class MyPolynomialRegressor:
    def __init__(self, degree=5):
        self.poly_reg = PolynomialFeatures(degree=degree)
        self.lin_reg = LinearRegression()

    def fit(self, X, y):
        X_poly = self.poly_reg.fit_transform(X)
        self.lin_reg.fit(X_poly, y)

    def predict(self, X):
        return self.lin_reg.predict(X=self.poly_reg.fit_transform(X))


def get_best_model_1_output(data):
    """

    :param data: a pandas data frame with the last column being the dependent variable
    :return: The best model for the data
    """
    # TODO: Perform backwards elimination for the best model

    # Identify data dimensions
    n_inputs = len(data.iloc[0, :].values) - 1
    sample_size = len(data.iloc[:, 0].values)

    regressors = [LinearRegression(),
                  MyPolynomialRegressor(degree=5),
                  SVR(kernel="rbf"),
                  DecisionTreeRegressor(random_state=42),
                  RandomForestRegressor(n_estimators=20, random_state=42)]

    best_model = None
    best_mse = 10000

    X = data.iloc[:, 0:n_inputs].values
    y = data.iloc[:, n_inputs].values

    # Feature scaling
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X = sc_x.fit_transform(X)
    y = sc_y.fit_transform(np.reshape(y, (-1, 1)))
    y = y.ravel()

    # Run each model
    for regressor in regressors:
        # initialize total mse
        # TODO: maybe add vmse variance of mse or use adjusted r2
        mmse_train = 0
        mmse_test = 0
        n_repeats = 20  # number of times the model is trained on different training and test sets
        test_size = 0.2  # proportion of the data that will be part of test set
        # Run each model with a new random state
        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            regressor.fit(X_train, y_train)

            mse_train = mean_squared_error(y_train, regressor.predict(X_train))
            mse_test = mean_squared_error(y_test, regressor.predict(X_test))
            mmse_train += mse_train
            mmse_test += mse_test

        mmse_train = mmse_train / n_repeats
        mmse_test = mmse_test / n_repeats
        mse = test_size * mmse_test + (1 - test_size) * mmse_train

        print("Done training, now testing to see if its better")
        # If the model is not over fitted and it has a better mse
        if (mmse_train < 1.1 * mmse_test) and mse < best_mse:
            best_mse = mse
            best_model = regressor

    plt.scatter(X[:, 0], y, color="red")
    plt.plot(X[:, 0], best_model.predict(X), color="blue")
    plt.xlabel("Level")
    plt.ylabel("Salary")
    plt.title("HI")
    plt.show()

    return best_model


def impute(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(data.iloc[:, :].values)
    data = imputer.transform(data)
    return pd.DataFrame(data)


def remove_english(data):
    data = pd.DataFrame(data)
    print(data.dtypes)
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


def get_best_model(data, n_outputs):
    data = remove_english(data)
    print(data)
    data = impute(data)
    models = []
    n_inputs = len(data.iloc[0, :]) - n_outputs
    # For each output, find the best model from the previous
    for i in range(n_outputs):
        models.append(get_best_model_1_output(data.iloc[:, :n_inputs + i + 1]))
    return models


if __name__ == "__main__":
    start_time = time.time()
    dataset = pd.read_csv("Position_Salaries.csv")
    # dataset = dataset.iloc[:, 1:3]
    print(type(dataset))
    print(get_best_model(dataset, 1))
    end_time = time.time()
    print("Time taken:", end_time - start_time)
