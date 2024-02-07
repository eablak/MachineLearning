# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:09:15 2024

@author: esraablak
"""

# METODOLOJİ: CRISP-DM !!!!

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

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,
                                        train_size=0.33,random_state=0)

"******************************************************************************"

"""
X tarafında verilerin değerleri birbirlerinden çok farklıdır. yaş boy kilo değerlerine baktığımızda kimisi 30-40 ağırlıklı değerlerken kimisi 170-180 civarında olabiliyor ve biz bu değerler arasında mantıklı bir ilişki kurarak cinsiyeti tahmin etmeye çalışıyoruz. Veriler birbirlerinden bu kadar farklıyken bu durumu engellemek için standardScaler ile veri setindeki özelliklerin (features) ortalamasını 0 ve standart sapmasını 1 yaparak özellikleri standart normak dağılıma dönüştürür. (mean = 0, standart deviation = 1)
"""


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)






















