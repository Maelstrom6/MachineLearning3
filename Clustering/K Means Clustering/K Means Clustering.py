import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# import the data set
dataset = pd.read_csv("Mall_Customers.csv")

# make sure X is a matrix not a vector
X = dataset.iloc[:, [3, 4]].values

# Using elbow method to find clusters
from sklearn.cluster import KMeans
wcss_array = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    k_means.fit(X)
    wcss_array.append(k_means.inertia_)

adjusted_wcss_array = [(i+1)**2*x for i, x in enumerate(wcss_array)]
print(adjusted_wcss_array)
gradients_of_adjusted_wcss_array = [adjusted_wcss_array[i + 1]-adjusted_wcss_array[i]
                                    for i in range(len(adjusted_wcss_array) - 1)]

min_index, min_value = 0, gradients_of_adjusted_wcss_array[0]
for i, x in enumerate(gradients_of_adjusted_wcss_array):
    if x < min_value:
        min_index, min_value = i, x


array = [x/max(wcss_array) for x in wcss_array]
plt.plot(np.arange(1, 11), array)
array = [x/max(adjusted_wcss_array) for x in adjusted_wcss_array]
plt.plot(np.arange(1, 11), array)
plt.title("Elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()
print(min_index + 2)


kmeans = KMeans(n_clusters=min_index + 2, init="k-means++", n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualising clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="green", label="Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="blue", label="Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="orange", label="Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c="magenta", label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, label="Centres")
plt.legend()
plt.show()

