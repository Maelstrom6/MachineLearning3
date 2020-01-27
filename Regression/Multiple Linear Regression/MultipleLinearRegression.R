# Title     : TODO
# Objective : TODO
# Created by: Chris
# Created on: 2019/07/16

dataset = read.csv("50_Startups.csv")

# Encoding catergorical data
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1, 2, 3))

# Split dataset into test set and training set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# regressor = lm(formula=Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = training_set)
regressor = lm(formula= Profit ~ ., data = training_set)

y_pred = predict(regressor, test_set)

# Building backward elimination
regressor = lm(formula=Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)

