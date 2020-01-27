# Title     : TODO
# Objective : TODO
# Created by: Chris
# Created on: 2019/07/16

dataset = read.csv("Data.csv")

# Taking care of missing data
# columns are case sensitive
dataset$Age = ifelse(is.na(dataset$Age),
ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
dataset$Salary)

# Encoding catergorical data
dataset$Country = factor(dataset$Country,
levels = c("France", "Spain", "Germany"),
labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
levels = c("No", "Yes"),
labels = c(0, 1))

# Split dataset into test set and training set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# If you get the error 'x' must be numeric
# then you need all your columns to be numbers
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])


print(training_set)
# print(dataset)
