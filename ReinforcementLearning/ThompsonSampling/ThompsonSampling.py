import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import math
import random

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000  # number of rounds to play
d = 10  # number of different ads
r0_i = [0] * d  # sum of punishments of ad i up to round n
r1_i = [0] * d  # sum of rewards of ad i up to round n
ads_selected = []

for n in range(N):
    random_betas = []
    for i in range(d):
        random_betas.append(random.betavariate(r1_i[i] + 1, r0_i[i] + 1))

    max_index, max_random = 0, random_betas[0]
    for i, upper_bound in enumerate(random_betas):
        if max_random < upper_bound:
            max_index, max_random = i, upper_bound

    result = dataset.iloc[n, max_index]
    r0_i[max_index] += 1 - result
    r1_i[max_index] += result
    ads_selected.append(max_index)


print(sum(r1_i))

# plt.hist(ads_selected, bins=d)
# plt.title("Ads VS the number of times they were chosen")
# plt.xlabel("Ad ID")
# plt.ylabel("Number of times chosen")
# plt.show()

plt.bar(np.arange(d), [(x+y) for x, y in zip(r1_i, r0_i)])
plt.title("Ads VS the number of times they were chosen")
plt.xlabel("Ad ID")
plt.ylabel("Number of times chosen")
plt.show()

plt.bar(np.arange(d), [x / (x+y) for x, y in zip(r1_i, r0_i)])
plt.title("Ads VS their sample probabilities")
plt.xlabel("Ad ID")
plt.ylabel("Sample probability of success")
plt.show()
