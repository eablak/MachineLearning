# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:22:15 2024

@author: esraablak
"""

"""
R2 (R-Square, R-Kare) Yöntemi

Buradaki amaç biz regresyon modelleri ile bir tahmin yapıyoruz ve çıktılar üretiyoruz. Bulduğumuz çıktılar gerçek değerden ne kadar uzak? Yani bizim modelimiz ne kadar doğru sonuçlar veriyor? Bir modele ne kadar güvenebilir? bunu r2 ile hesaplıyoruz. Modelimiz %100 doğru çalışırsa örneğin yaş hesaplıyosun tahmin 10 gerçek 10, tahmin 20 gerçek 20 bu böyle giderse grafiğini çizdiğinde doğrusal bir doğru çizilir. Hiçbir zaman %100 doğru tahminler yapamazsın o yüzden buradaki amaç bu doğrudan ne kadar saptığın onu hesaplamak.

1) Hata Kareleri Toplamı (Mean Squared Error (MSE)) = Topla(y - y')^2
2) Ortalama Farkların Toplamı (Sum of Squared Differences from the Mean (SSD)) = Topla(y - y(ort))^2
        R^2 = 1 - (HKT / OFT)        

R^2 -> (-) değerler alabilir ama bu çok çok kötü bi model demek yani yapamazsın bile
        0 çıkabilir. bu en kötü algoritmadır.             
        1 en iyi model demek yani fark yok / hata yok demek
        
        
R^2 için bir problem var yeni bir değşiken eklendiğinde iki ihtimal var
    - yeni eklenen değişken olumlu etki yapıyordur ve olumlu etki sonucunda R^2 değeri artıyordur
    - ya da olumsuz etki yapıyordur bu durumda yeni çarpanlar 0'a yakın oluyor (hatta 0 olup) hiçbir etki yapmayacak
dolayısıyla R^2 hesaplaması yaparken yeni değişkenler eklendiğinde sisteme asla negatif etki yapmaz, R^2 değerini asla düşürmez. Bunun içinse Düzeltilmiş R^2 (Adjusted R^2) Yöntemi uygulanır.
    Düzeltilmiş R^2 = 1 - (1 - R^2) * ((n-1) / (n - p - 1))
    p (yeni eklenen değişken sayısı)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler = pd.read_csv(r"C:\Users\ESRA  ABLAK\Desktop\MachineLearning\Regression\PolynomialRegression\maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X, Y, color="red")
plt.plot(x,lin_reg.predict(X), color="blue")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
plt.show()

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()


print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


"SVR outliner verilere karşı hassas bu yüzden scaler kullanılması gerek"

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_olcekli = sc.fit_transform(X)
y_olcekli = np.ravel(sc.fit_transform(Y.reshape(-1,1))) # ravel çok boyutlu dizileri tek boyutlu dizi haline getirir.



from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z = X - 0.4
K = X + 0.5

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")

plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

rf_reg.fit(X,Y.ravel()) # dataframe to np.array

#print("!",type(y))
#print("!",type(Y.ravel()))

print(rf_reg.predict([[6.6]]))


plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(X,rf_reg.predict(K),color="yellow")


"******************************************************************************************************************************"


from sklearn.metrics import r2_score

print("random forest ",r2_score(Y, rf_reg.predict(X)))
print("decision tree ",r2_score(Y, r_dt.predict(X))) # çıktısı 1 çünkü decision tree mevcut verilerden çıktı üretir yani r2'si 1 çıktı diye bu en iyi yöntem demek değil!
print("svr ",r2_score(y_olcekli,svr_reg.predict(x_olcekli)))
print("polynomial reg ", r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))
print("linear reg ", r2_score(Y,lin_reg.predict(X)))





















