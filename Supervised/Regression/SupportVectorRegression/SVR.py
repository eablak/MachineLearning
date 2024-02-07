# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:21:10 2024

@author: esraablak
"""

"""
İlk başta sınıflandırma için kullanılmış. Mantık şu iki veri kümesi içerisinde birbirlerine olan mesafeyi en uzun tutacak şekilde ayırmak. Otoyol mantığı verileri iki tarafa ayır ve en geniş aralığı bul. Regresyona geçildiğinde maxsimum noktayı(veri noktası) alabilen marjin aralığını bulmaya çalışır. Amaç marjin değerini minimize eden doğruyu bulmak?

SVR marjin değerlerini tanımlıyor ve bu marjin değerlerine giren maksimum noktayı elde edebileceği minimum marjin değerine sahip fonksiyonu almayı amaçlıyor. Birden fazla marjin doğrusu çizilebiliyorsa aynı noktaları içine alabilecek minimum marjin değerine sahip marjini elde etmeye çalışıyor. Bunun için farklı yöntemler kullanılıyor.
    - Doğrusal SVR
    - Doğrusal olmayan SVR (non-linear) 
        * Polynomial
        * Gaussian Radial Basis function
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

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))










