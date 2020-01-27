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

def get_distance(point1, point2):

    return math.sqrt(sum([(x-y)**2 for x, y in zip(point1, point2)]))


def get_dist_of_closest_center(point, centers):
    dist = get_distance(point, centers[0])
    for center in centers:
        dist_temp = get_distance(point, center)
        if dist_temp < dist:
            dist = dist_temp
    return dist


# Using elbow method to find clusters
from sklearn.cluster import KMeans
wcss_array = []
mad_array = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    k_means.fit(X)
    wcss_array.append(k_means.inertia_)
    total2 = 0
    centers = k_means.cluster_centers_
    for j, point in enumerate(X):
        temp = get_dist_of_closest_center(point, centers)
        total2 += temp
    mad_array.append(total2)




array = [x/max(wcss_array) for x in wcss_array]
plt.plot(np.arange(1, 11), array, label="WCSS")

adjusted_wcss_array = [(i+1)*x for i, x in enumerate(mad_array)]
array = [x/max(adjusted_wcss_array) for x in adjusted_wcss_array]
plt.plot(np.arange(1, 11), array, label="adjusted WCSS")

array = [x/max(mad_array) for x in mad_array]
plt.plot(np.arange(1, 11), array, label="MAD")

adjusted_mad_array = [(i+1)*x for i, x in enumerate(mad_array)]
array = [x/max(adjusted_mad_array) for x in adjusted_mad_array]
plt.plot(np.arange(1, 11), array, label="adjusted MAD")
plt.title("Elbow method MAD")
plt.xlabel("number of clusters")
plt.ylabel("MAD")
plt.legend()
plt.show()

difference = [x-y for x, y in zip(mad_array, wcss_array)]
array = [x/max(difference) for x in difference]
plt.plot(np.arange(1, 11), array, label="Difference")
plt.title("Elbow method MAD")
plt.xlabel("number of clusters")
plt.ylabel("MAD")
plt.legend()
plt.show()


gradients = []
for i in range(len(wcss_array) - 1):
    gradients.append(wcss_array[i + 1]-wcss_array[i])

angles = []
for i in range(len(wcss_array) - 2):
    # pi/2 + |theta| + pi/2 - |tau|
    tau = abs(math.atan(gradients[i]))
    theta = abs(math.atan(gradients[i + 1]))
    angles.append(math.pi + theta - tau)

print(gradients)
print(angles)

plt.plot(np.arange(2, len(wcss_array)), angles)
plt.title("Elbow method")
plt.show()


k_means = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300, random_state=0)
k_means.fit(X)
print(k_means.inertia_)
total1 = 0
total2 = 0
centers = k_means.cluster_centers_
for j, point in enumerate(X):
    temp = get_dist_of_closest_center(point, centers)
    total2 += temp
    total1 += temp**2
print(total1)
print(total2)


