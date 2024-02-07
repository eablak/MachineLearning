# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:48:58 2024

@author: esraablak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv(r"C:\Users\ESRA  ABLAK\Desktop\MachineLearning\VeriAnalizi\veriler.csv")

x = veriler.iloc[:,1:4].values # [5:,1:4]
y = veriler.iloc[:,-1].values # [5:,-1] olarak çalıştırınca (outlines) atınca doğru sınıflandırma yapıyor

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

from sklearn.svm import SVC

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)


"***************************************************************************************************"

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)





















