import random
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

file = open("creditcard.csv", "r")
filew = open("creditpositive.csv", "w")

randomNumbers = []
for i in range(500):
    randomNumbers.append(random.randint(1,284800))

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
    elif count in randomNumbers:
        filew.write(",".join(ls))
df = pd.read_csv('creditpositive.csv')

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

fig, (axis2,axis ) = plt.subplots(1,2)

axis.plot(x_test[y_predict == 0, 1], x_test[y_predict == 0, 29], '+b')
axis.plot(x_test[y_predict == 1, 1], x_test[y_predict == 1, 29], '+g')

axis2.set_title("Original data with classes.")
axis2.plot(x_test[y_test == 0, 1], x_test[y_test == 0, 29], '+b')
axis2.plot(x_test[y_test == 1, 1], x_test[y_test == 1, 29], '+g')

plt.show()
fig.savefig("OutlierDetection.pdf", dpi=None,
            facecolor='w', edgecolor='w', format="pdf")

req = LogisticRegression().fit(x_train,y_train)

y_predict = req.predict(x_test)

print("Accuracy score on balanced data with logistic regression "+str(accuracy_score(y_test,y_predict)))
fig2, (axis3,axis4 ) = plt.subplots(1,2)

axis4.set_title("logistic regresyon applied")
axis4.plot(x_test[y_predict == 0, 1], x_test[y_predict == 0, 29], '+b')
axis4.plot(x_test[y_predict == 1, 1], x_test[y_predict == 1, 29], '+g')

axis3.set_title("Original data with classes.")
axis3.plot(x_test[y_test == 0, 1], x_test[y_test == 0, 29], '+b')
axis3.plot(x_test[y_test == 1, 1], x_test[y_test == 1, 29], '+g')

plt.show()