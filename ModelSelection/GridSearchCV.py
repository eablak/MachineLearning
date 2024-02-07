# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:39:43 2024

@author: esraablak
"""

"""
Hiperparametre optimizasyonu => GridSearchCV, RandomizedCV

GridSearchCV: Modelde denenmesi istenen hiperparamtereler ve değerleri için bütün kombinasyonlar ile ayrı ayrı model kurulur ve belirtilen metriğe göre en başarılı hiperparametre seti belirlenir.
    (+) Tüm kombinasyonlar denendiği için en iyi performans gösteren hiperparametre setini belirlemeyi garanti eder. Küçük veri setlerinde ve sadece    birkaç tane hiperparametre denenmek istendiğinde çok iyi çalışır.
    (-) Büyük bir veri seti ile çalışıldığında ya da denenecek olan hiperparametre sayısı ve değeri arttırıldığında kombinasyon sayısı da artacaktır. Kurulan her modelin cross-validation ile test edildiği de düşünüldüğünde maliyet çok çok artar bu sebeple alternatif olarak RandomSearchCV yöntemi tercih edilebilir.
    
RandomizedSearchCV:
Rasgele olarak hiperparametre seti seçilir ve cross-validation ile model kurularak test edilir.Belirlenen hesaplama süresi limitine ya da iterasyon sayısna ulaşıncaya kadar bu adımlar devam eder.
    (+) Büyük veri setlerinde daha az maliyetle GridSearh yöntemiyle elde edilen en iyi skora yakın performans gösterecek hiperparametre setlerini belirleyebilir.
    (+) Daha geniş bit hiperparametre alanı tarayabilir.
    (-) Her ne kadar optimum hiperparametre setine yaklaşsa da tüm olası kombinasyonları tek tek denemediği için en iyi performans gösteren hiperparametre setini bulmayı garanti edemez.
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

"***********************************************************************************************************"

from sklearn.model_selection import GridSearchCV

p = [{'C': [1,2,3,4,5], 'kernel': ['linear']},
     {'C': [1,10,100,1000], 'kernel' : ['rbf'], 'gamma': [1,0.5,0.1,0.01,0.001]}]

gs = GridSearchCV(estimator = classifier, param_grid = p, scoring = "accuracy", cv = 10, n_jobs = -1)

grid_search = gs.fit(X_train, y_train)
best_score = grid_search.best_score_
best_params = grid_search.best_params_

print(best_score)
print(best_params)