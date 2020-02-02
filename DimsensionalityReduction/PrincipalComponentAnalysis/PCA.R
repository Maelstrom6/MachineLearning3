dataset = read.csv("Wine.csv")
# split into test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
train = subset(dataset, split==TRUE)
test = subset(dataset, split==FALSE)

# Feature scaling
train[, 1:13] = scale(train[, 1:13])
test[, 1:13] = scale(test[, 1:13])

# PCA
#install.packages("caret")
library(caret)
library(e1071)
pca = preProcess(x = train[-14], 
                 method = "pca", 
                 pcaComp = 2)  # thresh = 0.6
train = predict(pca, train)
# Swap the first and last column because the first column is now the dependent variable
train = train[, c(2, 3, 1)]

test = predict(pca, test)
test = test[, c(2, 3, 1)]

# Fitting model
#classifier = glm(formula=Customer_Segment ~ ., family=poisson, data=train)
classifier = svm(formula = Customer_Segment ~ ., 
                 data = train, 
                 type = "C-classification", 
                 kernel = "linear")

# Predicing
#p_pred = predict(classifier, type="response", newdata = test[, 1:2])
#y_pred = ifelse(p_pred < 1.5, 1, ifelse(p_pred < 2.5, 2, 3))
y_pred = predict(classifier, newdata = test[-3])

# Create confusion matrix
cm = table(test[,3], y_pred)

# Visualising results
# install.packages("ElemStatLearn")
library(ElemStatLearn)
set = test
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by=0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by=0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c("PC1", "PC2")
#prob_set = predict(classifier, type="response", newdata = grid_set)
#y_grid = ifelse(prob_set < 1.5, 1, ifelse(prob_set < 2.5, 2, 3))
y_grid = predict(classifier, type="response", newdata = grid_set)

plot(set[, -3],
     main = "Logistic Regression",
     xlab = "PC1",
     ylab = "PC2",
     xlim = range(X1),
     ylim = range(X2))

contour(X1, X2, 
        matrix(as.numeric(y_grid), length(X1), length(X2)),
        add=TRUE)

points(grid_set, pch=".", col=ifelse(y_grid == 1, "springgreen3", ifelse(y_grid == 2, "tomato", "blue2")))
points(set, pch=21, bg=ifelse(set[, 3] == 1, "green4", ifelse(set[, 3] == 2, "red3", "blue1")))





