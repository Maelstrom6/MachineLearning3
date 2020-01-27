dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[, 3:5]
# split into test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, 0.75)
train = subset(dataset, split==TRUE)
test = subset(dataset, split==FALSE)

# Feature scaling
train[, 1:2] = scale(train[, 1:2])
test[, 1:2] = scale(test[, 1:2])

# Fitting model and predicting
#install.packages("class")
# -3 removes the third column
library(class)
y_pred = knn(train[, -3], 
             test[, -3], 
             train[, 3],
             k = 5)

# Create confusion matrix
cm = table(test[,3], y_pred)

# Visualising results
# install.packages("ElemStatLearn")
library(ElemStatLearn)
set = train
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by=0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by=0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c("Age", "EstimatedSalary")
y_grid = knn(train[, -3], 
             grid_set, 
             train[, 3],
             k = 5)

plot(set[, -3],
     main = "KNN Regression",
     xlab = "Age",
     ylab = "Salary",
     xlim = range(X1),
     ylim = range(X2))

contour(X1, X2, 
        matrix(as.numeric(y_grid), length(X1), length(X2)),
        add=TRUE)

points(grid_set, pch=".", col=ifelse(y_grid == 1, "springgreen3", "tomato"))
points(set, pch=21, bg=ifelse(set[, 3] == 1, "green4", "red3"))





