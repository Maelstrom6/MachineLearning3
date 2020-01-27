import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# import the data set
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

regressor = RandomForestRegressor(n_estimators=10, random_state=42)
regressor.fit(x, y)

# y_pred = sc_y.inverse_transform(regressor.predict(sc_x.fit_transform([[6.5]])))
y_pred = regressor.predict([[6.5]])
print(y_pred)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("Hi")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

