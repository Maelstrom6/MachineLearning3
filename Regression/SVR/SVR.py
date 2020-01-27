import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# import the data set
dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature scaling. This SVR doesn't use feature scaling in its model
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(np.reshape(y, (-1, 1)))

regressor = SVR(kernel="rbf")  # Gaussian kernel
regressor.fit(x, y)

# y_pred = regressor.predict(sc_x.transform([[6.5]]))
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
print(y_pred)

plt.scatter(x, y, color="red")
plt.plot(x, regressor.predict(x), color="blue")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("HI")
plt.show()
