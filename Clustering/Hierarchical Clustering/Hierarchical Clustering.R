dataset = read.csv("Mall_Customers.csv")

X = dataset[, 4:5]

# Using dendrogram
set.seed(6)
dendrogram = hclust(dist(X, method = "euclidean"), method = "ward.D")
plot(dendrogram, 
     main = paste("Dendrogram"), 
     xlab = "Customers", 
     ylab = "Euclidean distances")

# Fitting model
hc = hclust(dist(X, method = "euclidean"), method = "ward.D")
y_hc = cutree(hc, 5)

# visualising clusters
library(cluster)
clusplot(X, y_hc, 
         lines = 0, 
         shade = TRUE, 
         color = TRUE, 
         labels = 2, 
         plotchar = FALSE, 
         span = TRUE, 
         main = paste("Clusters of clients"), 
         xlab = "Income", 
         ylab = "Spending")

