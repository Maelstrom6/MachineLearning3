import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# import the data set
dataset = pd.read_csv("Iris.csv")

# make sure X is a matrix not a vector
X = dataset.iloc[:, [1, 2, 3, 4]].values

# Using elbow method to find clusters
from sklearn.cluster import KMeans
wcss_array = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    k_means.fit(X)
    wcss_array.append(k_means.inertia_)

adjusted_wcss_array = [(i+1)*1/x for i, x in enumerate(wcss_array)]
adjusted_wcss_array = [x/max(adjusted_wcss_array) for x in adjusted_wcss_array]

gradients_of_adjusted_wcss_array = []
for i in range(len(adjusted_wcss_array) - 1):
    gradients_of_adjusted_wcss_array.append(adjusted_wcss_array[i + 1]-adjusted_wcss_array[i])

angles_of_adjusted_wcss_array = []
for i in range(len(wcss_array) - 2):
    # pi/2 + |theta| + pi/2 - |tau|
    # pi/2 + (pi/2 + tau) - theta
    tau = (math.atan(gradients_of_adjusted_wcss_array[i]))
    theta = (math.atan(gradients_of_adjusted_wcss_array[i + 1]))
    angles_of_adjusted_wcss_array.append(math.pi + tau - theta)

array = angles_of_adjusted_wcss_array
min_index, min_value = 0, array[0]
for i, x in enumerate(array):
    if x < min_value:
        min_index, min_value = i, x


array = [x/max(wcss_array) for x in wcss_array]
plt.plot(np.arange(1, 11), array, label="WCSS")
array = [x/max(adjusted_wcss_array) for x in adjusted_wcss_array]
plt.plot(np.arange(1, 11), array, label="Adjusted")
plt.title("Elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.legend()
plt.show()

plt.plot(np.arange(2, len(wcss_array)), angles_of_adjusted_wcss_array)
plt.title("Elbow method")
plt.show()

print(angles_of_adjusted_wcss_array)
print(min_index + 2)
n_clusters = min_index + 2
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualising clusters
color_array = ["red", "green", "blue", "orange", "magenta"]
for i in range(n_clusters):
    plt.scatter(X[y_kmeans == i, 0],
                X[y_kmeans == i, 1],
                s=100,
                c=color_array[i % len(color_array)],
                label="Cluster "+str(i+1))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, label="Centroids")
plt.legend()
plt.show()

