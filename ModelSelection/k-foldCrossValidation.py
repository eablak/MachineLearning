# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:32:47 2024

@author: esraablak
"""

"""
Cross validation n-fold: n = 4 olsun. Normalde veriyi genel olarak 1/3'ü test 2/3'ü eğitim olacak şekilde eğitirdik. cross validation için eğer n değeri dörtse; ilk önce  1/4 -> test 3/4 ->eğitim sonra, eğitim kümesinden bir bir kısım (total verinin 1/4'ü olacak şekilde) test kısmı kalanları eğitim, daha sonra kalan eğitim parçalarını sırayla test yapar ve diğerleri eğitim olur. Buradaki amaç veri setinin tamamını train ve test yapabilmek. Her bir test train eğitimi sırasında accuracy alır sonuç olarak da bu dört accuracy sonucunun ortalamasını alarak sonuç elde eder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = classifier, X = X_train, y= y_train, cv=4)
print(success.mean())