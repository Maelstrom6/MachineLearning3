
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# nltk.download("stopwords")

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Sanitise text
corpus = []
for review in dataset.iloc[:, 0].values:
    review = review.replace("n't", " not")
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = set(review.split())  # A set removes duplicate words and puts them in a set

    # Removes common words and find roots of remaining words
    ps = PorterStemmer()
    sw = set(stopwords.words("english"))
    sw.remove("not")
    sw.remove("no")
    review = [ps.stem(word) for word in review if word not in sw]
    review = " ".join(review)
    corpus.append(review)

# Creating bag of words model
cv = CountVectorizer(max_features=700)  # keep 1500 most frequent words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Common models = naive bayes and trees
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting logistic regression model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=42)  # 0.63
# classifier = GaussianNB()  # 0.67
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))

