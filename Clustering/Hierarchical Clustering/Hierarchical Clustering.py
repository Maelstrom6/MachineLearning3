
# Agglomerative clustering. The other option divisive
# 4 choices for distances between clusters

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:, [3, 4]].values

# using dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
heights = dendrogram["dcoord"]  # The set of quads of height values paired up with `icoord` values
flat_list = list(set([item for sublist in heights for item in sublist]))
flat_list.sort()  # A list of distinct height coordinates for the dendrogram
height_of_max_dist, max_dist = 0, 0
for i in range(len(flat_list) - 1):
    if max_dist < flat_list[i + 1] - flat_list[i]:
        max_dist = flat_list[i + 1] - flat_list[i]
        height_of_max_dist = flat_list[i] + max_dist/2  # pick the midpoint height to be safe

verticals = []  # Array of vertical lines that form the vertical lines of dendrogram
for quad in heights:
    verticals.append(quad[0:2])
    verticals.append(quad[2:4])

# count how many verticals cross through height_of_max_dist
optimal_n_clusters = 0
for vertical in verticals:
    if min(vertical) < height_of_max_dist < max(vertical):
        optimal_n_clusters += 1

# Showing the dendrogram
plt.title("Dendrogram")
plt.xlabel("Points")
plt.ylabel("Distances")
plt.show()

# Fitting clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=optimal_n_clusters, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

# visualising clusters
color_array = ["red", "green", "blue", "orange", "magenta"]
for i in range(optimal_n_clusters):
    plt.scatter(X[y_hc == i, 0],
                X[y_hc == i, 1],
                s=100,
                c=color_array[i % len(color_array)],
                label="Cluster " + str(i + 1))
plt.legend()
plt.show()


