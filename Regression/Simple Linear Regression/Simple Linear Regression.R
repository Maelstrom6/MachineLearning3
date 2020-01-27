
dataset = read.csv("Salary_Data.csv")

# install.packages("caTools")
library(caTools)
set.seed(123)

# split the dataset
split = sample.split(dataset$Salary, SplitRatio = 2/3)
trainingset = subset(dataset, split == TRUE)
testset = subset(dataset, split == F)

# Fitting the 
regressor = lm(formula = Salary ~ YearsExperience, data = trainingset)

# Predicing test set results
y_pred = predict(regressor, newdata = testset)

# Visualising results
# install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = trainingset$YearsExperience, y = trainingset$Salary),
             color = "red") +
  geom_line(aes(x = trainingset$YearsExperience, y = predict(regressor, newdata = trainingset)),
            color = "blue") +
  ggtitle("Salary VS Experience (Training set)") +
  xlab("Years of Experience") +
  ylab("Salary")

ggplot() +
  geom_point(aes(x = testset$YearsExperience, y = testset$Salary),
             color = "red") +
  geom_line(aes(x = trainingset$YearsExperience, y = predict(regressor, newdata = trainingset)),
            color = "blue") +
  ggtitle("Salary VS Experience (Test set)") +
  xlab("Years of Experience") +
  ylab("Salary")

