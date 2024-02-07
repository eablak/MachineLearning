# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:52:11 2024

@author: esraablak
"""

"""
Simple Linear Regression: y = ax+b ile doğru çizme ve veriler üzerinde uygulamak.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("satislar.csv")
#print(veriler)

aylar = veriler[["Aylar"]] # bağımsız değişken
satislar = veriler[["Satislar"]] # bağımlı değiken

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar,
                                    test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

"""
fit_transform(): Bu fonksiyon, veri setinin ölçeklendirme işlemi için kullanılan parametreleri hesaplar ve bu parametreleri eğitim verisi üzerine uygular. Yani, bu işlem hem veriye uyum sağlar (fit), hem de veriyi dönüştürür (transform).
transform(): Sadece ölçeklendirme parametrelerini kullanarak veriyi dönüştürür. Eğitim verisindeki hesaplanan parametreleri kullanarak, test verisini aynı ölçekleme kurallarıyla dönüştürür.
"""

Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(X_test) # kendi Y_test'ini oluşturuyo

"******************************************************************************"

x_train = x_train.sort_index() 
y_train = y_train.sort_index()
" indexlere göre sıralamazsan karışık olarak grafik çiziyo yani anlamsız bir grafik oluyor. "
plt.plot(x_train,y_train)

plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")






















