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

# KPCA
#install.packages("kernlab")
library(kernlab)
kpca = kpca(x = ~ ., data = train[-3], kernel = "rbfdot", features = 2)
train = as.data.frame(predict(kpca, train))
train$Purchased = subset(dataset$Purchased, split==TRUE)

test = as.data.frame(predict(kpca, test))
test$Purchased = subset(dataset$Purchased, split==FALSE)



# Fitting model
classifier = glm(formula=Purchased ~ ., family=binomial, data=train)

# Predicing
p_pred = predict(classifier, type="response", newdata = test[, 1:2])
y_pred = ifelse(p_pred > 0.5, 1, 0)

# Create confusion matrix
cm = table(test[,3], y_pred)

# Visualising results
# install.packages("ElemStatLearn")
library(ElemStatLearn)
set = train
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by=0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by=0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c("V1", "V2")
prob_set = predict(classifier, type="response", newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)

plot(set[, -3],
     main = "Logistic Regression",
     xlab = "V1",
     ylab = "V2",
     xlim = range(X1),
     ylim = range(X2))

contour(X1, X2, 
        matrix(as.numeric(y_grid), length(X1), length(X2)),
        add=TRUE)

points(grid_set, pch=".", col=ifelse(y_grid == 1, "springgreen3", "tomato"))
points(set, pch=21, bg=ifelse(set[, 3] == 1, "green4", "red3"))





