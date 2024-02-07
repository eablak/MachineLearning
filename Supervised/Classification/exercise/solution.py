# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:49:56 2024

@author: esraablak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_excel("Iris.xls")

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values
y = y.ravel()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logr = LogisticRegression(random_state=0)

logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.svm import SVC

svc = SVC(kernel="poly")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")

rfc.fit(X_train, y_train)
y_pred = rfc.predict(x_test)
y_proba = rfc.predict_proba(X_test) # hangi sınıftan olma olasılığı

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(y_proba)

# !ROC
from sklearn import metrics

fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,0], pos_label='e')
print(fpr)
print(tpr)

"---------------------------------------------------------------------------------------------------------"

from sklearn import datasets

iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

plt.show()
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.decomposition import PCA

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set_title("First three PCA dimensions")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()


