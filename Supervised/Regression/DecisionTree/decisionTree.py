# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:04:40 2024

@author: esraablak
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


"******************************************************************************************************************************"

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z = X - 0.4
K = X + 0.5

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")

plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))




