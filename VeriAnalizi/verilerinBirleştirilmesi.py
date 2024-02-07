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
#axis=1 verileri yan yana ekle, default: alt alta ekler (0)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)