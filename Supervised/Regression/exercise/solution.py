# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:27:30 2024

@author: esraablak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

veriler = pd.read_csv("maaslar_yeni.csv")

# preprocessing

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]

X = x.values
Y = y.values

#print(veriler.corr())

#independent = pd.DataFrame(data=X, index=range(30), columns=["unvanSeviye","kidem","puan"])
#dependent = pd.DataFrame(data=Y, index=range(30), columns=["maas"])

#!!!!! Linear Regresyon

lin_reg = LinearRegression()

lin_reg.fit(X,Y)

# kolonların p-değerine bak (p-value büyüdükçe h1 yanlış olma ihtimali artar) aynı zamanda R^2 değerleri de yazıyor


model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())


"""2 ve 3. kolonların p-değerleri yüksek çıkıyor o zaman bunların etki seviyesi düşük. 1. kolonla bi devam et
    x = veriler.iloc[:,2:5]  =>  x = veriler.iloc[:,2:3]   
böyle yapınca tabii R^2 değeri artıyor ama tek kolon üzerinden devam ediyorsun ?
"""
print("Linear R2 değeri")
print(r2_score(Y, lin_reg.predict(X)))



#!!!!! Polynomial Regression

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print("poly OLS")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print("polynomial r2")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


#!!!!! SVR
    # scaling
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

print("svr ols")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print("svr r2")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


#!!!!! Decision Tree
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print("dt ols")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print("dt r2")
print(r2_score(Y,r_dt.predict(X)))


#!!!!! Random Forest
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print("rf ols")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary)

print("rf r2")
print(r2_score(Y, rf_reg.predict(X)))


"x2 ve x3 kolonunu alıp r2 sonucuna baktığında düştüğünü görüyoruz yani daha az doğru sonuçlar verebiliyor tabii bunları karşılaştırman lazım ona göre seçiceksin bazı modellerde daha doğru sonuçlar verdiği de oluyor"

"""
    Tahmin Modelleri
    
Linear Regression 
    (+) Veri boyutundan bağımsız olarak doğrusal ilişki üzerine kurulur
    (-) Doğrusallık kabülü aynı zamanda hatalıdır
    
Polynomial Regression
    (+) Doğrusal olmayan problemleri adresler
    (-) Başarı için doğru polinom derecesi önemlidir.

SVR
    (+) Doğrusal olmayan modellerde çalışır, marjinal değerlere karşı ölçekleme ile dayanıklı olur
    (-) Ölçekleme önemlidir, anlaşılması nispeten karışıktır, doğru kernel fonksiyonu seçimi önemlidir.
    
DTR
    (+) anlaşılabilirdir, ölçeklemeye ihtiyaç duymaz. doğrusal veya doğrusal olmayan problemlerde çalışır.
    (-) sonuçlar sabitlenmiştir, küçük veri kümelerinde ezberleme olması yüksek ihtimallidir.

RF
    (+) anlaşılabilirdir, ölçeklemeye ihtiyaç duymaz. doğrusal veya doğrusal olmayan problemlerde çalışır, ezber ve sabit sonuç riski düşüktür.
    (-) çıktıların yorumu ve görsellemesi nispeten zordur
"""








