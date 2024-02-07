# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:09:15 2024

@author: esraablak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")

from sklearn.impute import SimpleImputer

Yas = veriler.iloc[:,1:4].values

imputer = SimpleImputer(missing_values=np.NaN, strategy="mean")
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])

ulke = veriler.iloc[:,0:1]

from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

"******************************************************************************"

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=["boy","kilo","yas"])
cinsiyet = veriler.iloc[:,-1].values
sonuc3= pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
s = pd.concat([sonuc,sonuc2], axis=1) 
s2 = pd.concat([s,sonuc3], axis=1)

"******************************************************************************"

"""
ülke, boy, kilo, yaştan cinsiyetin tahmin edilmesini istiyoruz. Dolayısıyla ülke, boy, kilo, yaş ayrı bir dataFrame'de cinsiyet ayrı bir dataFrame'de bölmeliyiz.
    X => bağımsız değişken (ülke, boy, kilo, yaş)
    y => bağımlı değişken (cinsiyet)
aynı zamanda veriyi ikiye böleceksin.
2 farklı boyutta bölme işlemi yapıyoruz. train ve test bölümü verinin satır bazlı bölümü. x ve y bağımsız ve bağımlı olarak bölünmesi. 
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(s,sonuc3,
                                        train_size=0.33,random_state=0)

# train_size = train'e ne kadar veri gideceği
# random_state = verilerin rastgele bölünmesi ve bu bölme işleminin tekrarlanabilmesi


























