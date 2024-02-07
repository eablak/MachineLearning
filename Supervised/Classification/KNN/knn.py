# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:58:55 2024

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
print(cm)

"***************************************************************************************************"

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski") #komşu sayısı için default 5 ama en verimli olanı sen seç
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)