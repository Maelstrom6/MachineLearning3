dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[, 3:5]

# Split data
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, 0.75)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

# Feature scaling
train[, 1:2] = scale(train[, 1:2])
test[, 1:2] = scale(test[, 1:2])

# Fitting the model
# install.packages("e1071")
library(e1071)
classifier = svm(formula = Purchased ~ ., 
                 data = train, 
                 type = "C-classification", 
                 kernel = "linear")

y_pred = predict(classifier, newdata = test[-3])



# Create confusion matrix
cm = table(test[,3], y_pred)

unscale <- function(z, center = attr(z, "scaled:center"), scale = attr(z, "scaled:scale")) {
  if(!is.null(scale))  z <- sweep(z, 2, scale, `*`)
  if(!is.null(center)) z <- sweep(z, 2, center, `+`)
  structure(z,
            "scaled:center"   = NULL,
            "scaled:scale"    = NULL,
            "unscaled:center" = center,
            "unscaled:scale"  = scale
  )
}

# Visualising results
# install.packages("ElemStatLearn")
library(ElemStatLearn)
set = train### new
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by=0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by=0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c("Age", "EstimatedSalary")
#grid_set = scale(grid_set) ### new
y_grid = predict(classifier, newdata = grid_set)

plot(set[, -3],
     main = "SVM Regression",
     xlab = "Age",
     ylab = "Salary",
     xlim = range(X1),
     ylim = range(X2))

contour(X1, 
        X2, 
        matrix(as.numeric(y_grid), length(X1), length(X2)),
        add=TRUE)

points(grid_set, pch=".", col=ifelse(y_grid == 1, "springgreen3", "tomato"))
points(set, pch=21, bg=ifelse(set[, 3] == 1, "green4", "red3"))


