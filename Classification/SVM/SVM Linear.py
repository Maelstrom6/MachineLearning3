import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# import the data set
dataset = pd.read_csv("Social_Network_Ads.csv")

# make sure X is a matrix not a vector
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting logistic regression model
from sklearn.svm import SVC
# classifier = SVC()  # The gaussian kernel
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from matplotlib.colors import ListedColormap
X_set = sc_X.inverse_transform(X_test)
y_set = y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step=((X_set[:, 0].max())-(X_set[:, 0].min()))/100),
                     np.arange(start=(X_set[:, 1].min()) - 1,
                               stop=(X_set[:, 1].max()) + 1,
                               step=((X_set[:, 1].max())-(X_set[:, 1].min()))/100))
plt.contourf(X1, X2,
             classifier.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                cmap=ListedColormap(("red", "green"))(i),
                label=j)
plt.ylabel("Salary")
plt.xlabel("Age")
plt.legend()
plt.show()

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min(), stop=X_set[:, 0].max(), step=0.01),
                     np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max(), step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                cmap=ListedColormap(("red", "green")))
plt.ylabel("Salary")
plt.xlabel("Age")
plt.legend()
plt.show()

