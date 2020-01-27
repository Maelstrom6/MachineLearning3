# Title     : TODO
# Objective : TODO
# Created by: Chris
# Created on: 2019/07/16

dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

# Create regressor
# install.packages("e1071")
library(e1071)
regressor = e1071::svm(formula = Salary ~ ., 
                       data = dataset, 
                       type = "eps-regression")

# Create the prediction
y_pred = predict(regressor, data.frame(Level = 6.5))
print(y_pred)

# Graoh
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), color = "blue") +
  ggtitle("Yes") + 
  xlab("Level") +
  ylab("Salary")




