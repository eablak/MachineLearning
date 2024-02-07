# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:17:15 2024

@author: esraablak
"""

"""
regresyonda destek vektörler arasındaki aralığa düşen örnek sayısını maksimize etmeye çalışıyoruz. sınıflandırma için ise bu aralığa hiçbir noktanın düşmemesi. yani ayrımı sağlayabilmek
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

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski") #komşu sayısı için default 5 ama en verimli olanı sen seç
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

"***************************************************************************************************"

from sklearn.svm import SVC

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

"""
    Kernel Trick
Elindeki veri seti linear bir şekilde ayrılamayacak şekildeyse svm nasıl kullanırsın? bunun için boyut arttırma veya indirgeme yaparsın. böylelikle non-linear bir marjin çizerek verileri ayırırsın.
    (Çoklu kernel) bazı veriler non linear yapı kurmak için de zordur (çok iç içe geçmiş veriler vs) bunlar için ise birden fazla noktadan çekersin (boyutlar/koni oluşturusun). öyle non-linear bir yapı kurarsın.
"""



















