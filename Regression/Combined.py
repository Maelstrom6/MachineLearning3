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
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from joblib import dump, load


class MyPolynomialRegressor:
    def __init__(self, degree=5):
        self.poly_reg = PolynomialFeatures(degree=degree)
        self.lin_reg = LinearRegression()

    def fit(self, X, y):
        X_poly = self.poly_reg.fit_transform(X)
        self.lin_reg.fit(X_poly, y)

    def predict(self, X):
        return self.lin_reg.predict(X=self.poly_reg.fit_transform(X))


def get_best_regressor(data):
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
    print(data)

    X = data.iloc[:, 0:n_inputs].values
    print(X)
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
        n_repeats = 50  # number of times the model is trained on different training and test sets
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
        print(mmse_train)
        print(mmse_test)
        print("-----------")
        mse = test_size * mmse_test + (1 - test_size) * mmse_train

        print("Done training, now testing to see if its better")
        # If the model is not over fitted and it has a better mse
        #if (mmse_train < 1.1 * mmse_test) and mse < best_mse:
        # TODO: Check this if statement
        if mse < best_mse:
            best_mse = mse
            # To get the best possible version, we must train it on all the data we have

            regressor.fit(X, y)
            best_model = regressor
    print(best_mse)
    print(mean_squared_error(y, regressor.predict(X)))

    plt.scatter(X[:, -1], y, color="red")
    plt.plot(X[:, -1], regressor.predict(X), color="blue")
    plt.xlabel("Level")
    plt.ylabel("Salary")
    plt.title("HI")
    plt.show()

    return best_model


def get_best_classifier(data):
    """

        :param data: a pandas data frame with the last column being the dependent variable
        :return: The best model for the data
        """
    # TODO: Perform backwards elimination for the best model

    # Identify data dimensions
    n_inputs = len(data.iloc[0, :].values) - 1
    sample_size = len(data.iloc[:, 0].values)

    classifiers = [#KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
                   DecisionTreeClassifier(criterion="entropy", random_state=42),
                   LogisticRegression(random_state=0),
                   GaussianNB(),
                   RandomForestClassifier(n_estimators=20, max_leaf_nodes=10, criterion="entropy", random_state=42),
                   #SVC(kernel="rbf", random_state=0, C=100)
                   ]

    best_model = None
    best_f1 = 0
    print(data)

    X = data.iloc[:, 0:n_inputs].values
    print(X)
    y = np.asarray(data.iloc[:, n_inputs].values, dtype=np.int)

    # Feature scaling
    sc_x = StandardScaler()
    X = sc_x.fit_transform(X)

    # Run each model
    for classifier in classifiers:
        # initialize total mse
        # TODO: maybe add vmse variance of mse or use adjusted r2
        mf1_train = 0
        mf1_test = 0
        n_repeats = 50  # number of times the model is trained on different training and test sets
        test_size = 0.2  # proportion of the data that will be part of test set
        # Run each model with a new random state
        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            classifier.fit(X_train, y_train)

            f1_train = f1_score(y_train, classifier.predict(X_train))
            f1_test = f1_score(y_test, classifier.predict(X_test))
            mf1_train += f1_train
            mf1_test += f1_test

        mf1_train = mf1_train / n_repeats
        mf1_test = mf1_test / n_repeats
        print(mf1_train)
        print(mf1_test)

        f1 = test_size * mf1_test + (1 - test_size) * mf1_train

        print("Done training, now testing to see if its better")
        print("-----------")
        # If the model is not over fitted and it has a better mse
        # if (mmse_train < 1.1 * mmse_test) and mse < best_mse:
        # TODO: Check this if statement
        if f1 > best_f1:
            best_f1 = f1
            # To get the best possible version, we must train it on all the data we have

            classifier.fit(X, y)
            best_model = classifier
    print(best_f1)
    print(f1_score(y, classifier.predict(X)))
    print(confusion_matrix(y, classifier.predict(X)))

    dump(X, "X.joblib")
    dump(y, "y.joblib")

    return best_model


def impute(data):
    """

    :param data: The pandas DataFrame to impute (only numbers allowed)
    :return: the imputed dataframe with the average in place of empty cells
    """
    # sklearn deletes the columns so we have to put them back later
    colun_names = data.columns
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(data.iloc[:, :])
    data = imputer.transform(data)
    data = pd.DataFrame(data, columns=colun_names)
    return data


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
    for i, item in enumerate(data.iloc[0, :].values):
        if type(item) == str:
            column_categories.append(1)
            data[data.columns[i]] = LabelEncoder().fit_transform(data[data.columns[i]])
        else:
            column_categories.append(0)
    return data#, column_categories


def get_best_model_dependent(data, n_outputs, output_types):
    """
    Calculates the best model for each output where the next output has all of the previous outputs as input too
    :param output_types: An array of 1s or 0s. 0 if regression problem and 1 if classification problem
    :param data: The pandas DataFrame to perform the modelling
    :param n_outputs: The number of dependent variables to model
    :return: An array of the best model for each output
    """
    n_inputs = len(data.iloc[0, :]) - n_outputs
    # For each output, find the best model from the previous
    models = []
    for i in range(n_outputs):
        # Separate dependent and independent variables for preprocessing
        # This stuff could go outside the for loop but as outputs become inputs, they get encoded differently

        X = data.iloc[:, 0:n_inputs + i]
        y = data.iloc[:, n_inputs + i]
        X = encode_one_hot(X)
        y = encode_labels(y)
        category = output_types[i]
        X = impute(X)
        y = impute(y)

        # Join them pack together
        current_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

        if category == 0:
            models.append(get_best_regressor(current_data))
        else:
            models.append(get_best_classifier(current_data))
    return models


def get_best_model_independent(data, n_outputs, output_types):
    """
    Calculates the best model for each output given the inputs where none of the outputs influence each other
    :param output_types: An array of 1s or 0s. 0 if regression problem and 1 if classification problem
    :param data: The pandas DataFrame to perform the modelling
    :param n_outputs: The number of dependent variables to model
    :return: An array of the best model for each output
    """
    n_inputs = len(data.iloc[0, :]) - n_outputs

    # Separate dependent and independent variables for preprocessing
    X = data.iloc[:, 0:n_inputs]
    Y = data.iloc[:, n_inputs:]

    X = encode_one_hot(X)
    Y = encode_labels(Y)

    X = impute(X)
    Y = impute(Y)

    # For each output, find the best model from the previous
    models = []
    for i in range(n_outputs):
        current_data = pd.concat([X.reset_index(drop=True), Y[Y.columns[i]].reset_index(drop=True)], axis=1)
        if output_types[i] == 0:
            models.append(get_best_regressor(current_data))
        else:
            models.append(get_best_classifier(current_data))
    return models


if __name__ == "__main__":

    start_time = time.time()
    dataset = pd.read_csv("Churn_Modelling.csv")
    dataset = dataset.iloc[:, 3:]
    models = get_best_model_dependent(dataset, 1, [1])
    dump(models, "Models.joblib")
    print(models)
    end_time = time.time()
    print("Time taken:", end_time - start_time)

    # models = load("Models.joblib")
    # X = load("X.joblib")
    # y = load("y.joblib")
    # print(models[0])
    # print(X)
    # print(y)
    #
    # print(f1_score(y,models[0].predict(X)))
    # cm = confusion_matrix(y,models[0].predict(X))
    # print(cm)
    # print((cm[0][0]+cm[1][1])/10000)


