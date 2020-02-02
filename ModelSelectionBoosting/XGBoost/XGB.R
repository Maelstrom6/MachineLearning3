dataset = read.csv("Churn_Modelling.csv")

dataset = dataset[, 4:14]



dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c("France", "Spain", "Germany"),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c("Male", "Female"),
                                   labels = c(1, 2)))
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting XGBoost
#install.packages("xgboost")
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)


y_pred = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = (y_pred >= 0.5)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# Applying kfold cv
library(caret)
folds = createFolds(y = training_set$Exited, k = 10)
cv = lapply(folds, FUN = function(x){
  train_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier_fold = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)
  y_pred_fold = predict(classifier_fold, newdata = as.matrix(test_fold[-11]))
  y_pred_fold = (y_pred_fold >= 0.5)
  cm_fold = table(test_fold[,11], y_pred_fold)
  print(cm_fold)
  accuracy_fold = (cm_fold[1, 1]+cm_fold[2, 2])/(cm_fold[1, 1]+cm_fold[2, 2]+cm_fold[1, 2]+cm_fold[2, 1])
  return(accuracy_fold)
  "training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)"
})

mean = mean(as.numeric(cv))
v = var(as.numeric(cv))

