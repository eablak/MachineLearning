# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:27:24 2024

@author: esraablak
"""

"""
Esemble Learning (Kollektif öğrenme): Birden fazla sınıflandırma algoritması ya da birden fazla tahmin algoritmasının aynı anda kullanılması. bu daha başarılı bir sonuç çıkartabilir. 
Random forest birden fazla decision tree'nin ayni veri kümesi için çizilmesi. Rassal orman bu farklı decision tree'leri alıyor ve majority vote işlemine tabii tutuyor. Birden fazla tahmin algoritması çalıştığı için çıkan sonuçları karşılaştırıyo. classsification ise çoğunluğun sonucunu nihai karar olarak veriyo regression ise ortalamayı alır sonucu o diye verir.
decision tree'lerde verilerin artması durumunda başarının düşmesi gibi bir sonuç var. çünkü çok fazla dallanma overfitting'e gider. aynı zamanda çok dallanma hesaplama maliyetini arttırır.
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

"******************************************************************************************************************************"

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

rf_reg.fit(X,Y.ravel()) # dataframe to np.array

#print("!",type(y))
#print("!",type(Y.ravel()))

print(rf_reg.predict([[6.6]]))
"yukarıda decision tree için 6.6 sayısını verdiğimizde çıktı 10000 çıkıyor bu kesin bi sonuç ama random forestte birden fazla sonuç oluştuğu için çıkan sonuçların ortalamasını döndürür. yani tahmin aşamasında orjinal veriler dışında veri döndürebiliyor. (tahminde farklı döndürebilir, sınıflandırmada mevcut olanlardan en çok çıkanı seçer yani mevcut verilerden farklı bişey seçemez)"

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(X,rf_reg.predict(K),color="yellow")










