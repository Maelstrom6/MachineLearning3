

dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

# Creating linear model
lin_reg = lm(formula = Salary ~ Level, data = dataset)

# Creating polynomial model
dataset$Level2 = dataset$Level ^ 2
dataset$Level3 = dataset$Level ^ 3
dataset$Level4 = dataset$Level ^ 4
poly_reg = lm(formula = Salary ~ ., data = dataset)

# Visualising data
# install.packages("ggplot2")
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = "red") + 
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = "blue") + 
  ggtitle("I am a title") +
  xlab("years") +
  ylab("salary")

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = "red") + 
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = "blue") + 
  ggtitle("I am a title") +
  xlab("years") +
  ylab("salary")              

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = "red") + 
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid, 
                                                                       Level2 = x_grid^2,
                                                                       Level3 = x_grid^3,
                                                                       Level4 = x_grid^4))), colour = "blue") + 
  ggtitle("I am a title") +
  xlab("years") +
  ylab("salary")      

# Predict individual response
y_pred = predict(lin_reg, data.frame(Level = 6.5))
print(y_pred)

y_pred = predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))
print(y_pred)



