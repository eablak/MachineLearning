# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:01:48 2024

@author: esraablak
"""

"""
Polinomal Regresyon
y = B0 + B1X + B2X^2 + B3^X3 + ... + BhX^h + e

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x,lin_reg.predict(x))

# polynomal regression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color="blue")
plt.show()

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color="pink")
plt.show()

# tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))


print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))







