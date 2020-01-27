import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# import the data set
dataset = pd.read_csv("Position_Salaries.csv")

# make sure X is a matrix not a vector
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)


# Fitting polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("hi i am a title")
plt.xlabel("years")
plt.ylabel("salary")
plt.show()

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color="red")
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("hi i am a title")
plt.xlabel("years")
plt.ylabel("salary")
plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))

