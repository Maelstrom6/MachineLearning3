
dataset = read.csv("Mall_Customers.csv")

X = dataset[, 4:5]

# Using elbow method
set.seed(6)
wcss = vector()

#for(i in 1:10){
#  wcss[i] = sum(kmeans(X, i)$withinss)
#}

for(i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)

plot(1:10, wcss, "b", 
     main = paste("Centres of clients"),
     xlab = "Number of clusters",
     ylab = "WCSS")

# Applying kmeans
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# visualising clusters
library(cluster)
clusplot(X, kmeans$cluster, 
         lines = 0, 
         shade = TRUE, 
         color = TRUE, 
         labels = 2, 
         plotchar = FALSE, 
         span = TRUE, 
         main = paste("Clusters of clients"), 
         xlab = "Income", 
         ylab = "Spending")


