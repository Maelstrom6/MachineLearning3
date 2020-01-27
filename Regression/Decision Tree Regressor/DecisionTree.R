# Title     : TODO
# Objective : TODO
# Created by: Chris
# Created on: 2019/07/16

dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

# Create regressor
# install.packages("rpart")
library(rpart)
regressor = rpart(formula = Salary ~ ., data = dataset, control = rpart.control(minsplit = 1))


# Create the prediction
y_pred = predict(regressor, data.frame(Level = 6.5))
print(y_pred)

# Graoh
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color = "blue") +
  ggtitle("Yes") + 
  xlab("Level") +
  ylab("Salary")




