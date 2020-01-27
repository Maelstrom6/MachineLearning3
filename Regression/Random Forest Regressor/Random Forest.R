# Title     : TODO
# Objective : TODO
# Created by: Chris
# Created on: 2019/07/16

dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

# Create regressor
# install.packages("Rtools")
# install.packages("randomForest")
library(randomForest)
# x gives a dataframe and y gives salary.
# Thats why there are 2 different syntaxes
set.seed(123)
regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 10)


# Create the prediction
y_pred = predict(regressor, data.frame(Level = 6.5))
print(y_pred)

# Graoh
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color = "blue") +
  ggtitle("Yes") + 
  xlab("Level") +
  ylab("Salary")




