import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import math

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000  # number of rounds to play
d = 10  # number of different ads
n_i = [0] * d  # number of times ad i was selected up to round n
r_i = [0] * d  # sum of rewards of ad i up to round n

for n in range(N):
    r_bar_i = []
    delta_i = []
    for i in range(d):
        if n_i[i] == 0:  # avoid division by zero error
            r_bar_i.append(0.5)  # a predefined assumption on the expected value
            delta_i.append(1000)
            # A large variance so it will be chosen above the others who have already run
        else:
            r_bar_i.append(r_i[i] / n_i[i])  # the sample proportion of rewards
            delta_i.append(math.sqrt(3 / 2 * math.log(n + 1) / n_i[i]))  # the sample variance of rewards

    upper_bounds = [x + y for x, y in zip(r_bar_i, delta_i)]

    max_index, max_bound = 0, upper_bounds[0]
    for i, upper_bound in enumerate(upper_bounds):
        if max_bound < upper_bound:
            max_index, max_bound = i, upper_bound

    result = dataset.iloc[n, max_index]
    n_i[max_index] += 1
    r_i[max_index] += result

print(n_i)
print(r_i)
print(sum(r_i))

plt.bar(np.arange(d), n_i)
plt.title("Ads VS the number of times they were chosen")
plt.xlabel("Ad ID")
plt.ylabel("Number of times chosen")
plt.show()

plt.bar(np.arange(d), [x / y for x, y in zip(r_i, n_i)])
plt.title("Ads VS their sample probabilities")
plt.xlabel("Ad ID")
plt.ylabel("Sample probability of success")
plt.show()
