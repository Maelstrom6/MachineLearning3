import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# Convert dataframe to 2D list
# transactions = dataset.values.tolist()
# print(transactions)

# transactions = []
# for i in range(7501):
#     inner_list = []
#     for j in range(20):
#         if str(dataset.iloc[i, j]) != 'nan':
#             inner_list.append(dataset.iloc[i, j])
#     transactions.append(inner_list)
# print(transactions)

transactions = []
for i in range(7501):
    transactions.append([str(dataset.iloc[i, j]) for j in range(20)])

from AccociationRuleLearning.AprioriARL.apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, max_length=3)  # min_length is also one

results = list(rules)
print(results[0])
