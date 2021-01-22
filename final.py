import random
import warnings
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
"""I didnt upload the original data to github since it's bigger than Git's size limit. Link to the data is here 
(https://www.kaggle.com/mlg-ulb/creditcardfraud) """
file = open("creditcard.csv", "r")
filew = open("creditpositive.csv", "w")

unbalanceddf = pd.read_csv('creditcard.csv')

"""
Data is highly unbalanced because of that I will be applying undersampling technique.

Firstly I attempted to do undersampling on imbalanced data.
500 random numbers will be selected randomly and these rows will be taken as undersampled.
"""

randomNumbers = []
for i in range(500):
    randomNumbers.append(random.randint(1, 284800))

randomNumbers.sort()

lspositive = []
count = 0
for line in file:
    count += 1

    if count == 1:
        filew.write(line)
        continue
    ls = line.split(",")
    if int(ls[-1][1]) == 1:

        filew.write(",".join(ls))
    elif count in randomNumbers:  # Random rows will be selected for undersampling
        filew.write(",".join(ls))
df = pd.read_csv('creditpositive.csv')
print("Data has been balanced by applying undersampling technique.")
ccfraud = df.to_numpy()

X = ccfraud[:, :-1]
Y = ccfraud[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
).fit(x_train, y_train)

y_predict = km.predict(x_test)
print(accuracy_score(y_test, y_predict))
"""I used only meaningful parameters for graphic plotting main reason to draw graphic is to see 
whether data is balanced and also determine performance of clustering algorithms  """
fig, (axis2, axis) = plt.subplots(1, 2)
axis.set_title("Clustering")
axis.plot(x_test[y_predict == 0, 1], x_test[y_predict == 0, 29], '+b')
axis.plot(x_test[y_predict == 1, 1], x_test[y_predict == 1, 29], '+g')

axis2.set_title("Original data with classes.")
axis2.plot(x_test[y_test == 0, 1], x_test[y_test == 0, 29], '+b')
axis2.plot(x_test[y_test == 1, 1], x_test[y_test == 1, 29], '+g')

fig.savefig("clustering on credit cart data.pdf", dpi=None,
            facecolor='w', edgecolor='w', format="pdf")

req = LogisticRegression(random_state=0).fit(x_train, y_train)

y_predict = req.predict(x_test)

print("Accuracy score on balanced data with logistic regression " + str(accuracy_score(y_test, y_predict)))
fig2, (axis3, axis4) = plt.subplots(1, 2)

axis4.set_title("logistic regresyon applied")
axis4.plot(x_test[y_predict == 0, 1], x_test[y_predict == 0, 29], '+b')
axis4.plot(x_test[y_predict == 1, 1], x_test[y_predict == 1, 29], '+g')

axis3.set_title("Original data with classes.")
axis3.plot(x_test[y_test == 0, 1], x_test[y_test == 0, 29], '+b')
axis3.plot(x_test[y_test == 1, 1], x_test[y_test == 1, 29], '+g')

clf = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)

y_predict = clf.predict(x_test)

print("Accuracy score for Random Forest classiffier: " + str(accuracy_score(y_test, y_predict)))
print("Data has been balanced with Smote")
unbalancedNumpy = unbalanceddf.to_numpy()

X = unbalancedNumpy[:, :-1]
Y = unbalancedNumpy[:, -1]
"""I applied smote for minority class generation because majority classes """
oversample = SMOTE()

x, y = oversample.fit_resample(X, Y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
).fit(x_train, y_train)

y_predict = km.predict(x_test)

print("Accuracy score for Kmeans: " + str(accuracy_score(y_test, y_predict)))

breq = LogisticRegression(random_state=0).fit(x_train, y_train)

y_predict = breq.predict(x_test)

print("Accuracy score for Logistic regression: " + str(accuracy_score(y_test, y_predict)))

fig3, (axis1, axis2) = plt.subplots(1, 2)

axis2.set_title("Logistic regresyon applied")
axis2.plot(x_test[y_predict == 0, 1], x_test[y_predict == 0, 29], '+b')
axis2.plot(x_test[y_predict == 1, 1], x_test[y_predict == 1, 29], '+g')

axis1.set_title("Original data with classes.")
axis1.plot(x_test[y_test == 0, 1], x_test[y_test == 0, 29], '+b')
axis1.plot(x_test[y_test == 1, 1], x_test[y_test == 1, 29], '+g')

plt.show()
