
# dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

# install.packages("arules")
library(arules)
# create a sparse matrix
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=50)

# support of i = # transactions containing i / total # transactions
# confidence of i to j = # transactions containing i and j /  # transactions containing i
# lift = confidence / support
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

inspect(sort(rules, by = "lift")[1:10])
