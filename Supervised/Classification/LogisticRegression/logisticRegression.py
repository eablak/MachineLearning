# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 08:43:36 2024

@author: esraablak
"""

"""
Logistic regresyonda bir doğru da çizilebilir. Sigmoid veya step function (adım fonksiyonu) gibi daha belirgin ayrımlarla sınıflandırılabilir.
Sigmoid -> S harfi gibi 2 kümeye ayırır (erkek/kadın) (evli/bekar) vs
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

"""

Confusion Matrix (Karmaşıklık Matrisi)
               C1(tahmin)        C2(tahmin)
C1(gerçek)   True positive    False negative
C2(gerçek)   False positive   True negative

Accuracy M, acc(M): model M için yüzde kaç doğru sınıflandırma yapar
    Error rate (misclassification rate) = 1 - acc(M)
    Alternatif ölçümler (e.g., for cancer diagnosis)
    sensitivity = t-pos / (t-pos+f-neg)    => true positive recognation rate
    specificity = t-neg / (t-neg+f-pos)    => true negative recognation rate
    precision = t-pos / (t-pos+f-pos)
    accuracy = sensitivity * pos/(pos + neg) + specificity * neg/(pos+neg)
"""

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)








