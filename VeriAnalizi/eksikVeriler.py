# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:27:30 2024

@author: esraablak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")
#print(veriler)

"""
Bazı makine öğrenme algoritmaları eksik verilerle çalışamıyorlar. Bunun önlemek için farklı yöntemlere başvurulur. Sayısal veriler için genellikle ortalama alınır. Alınan bu değer eksik verilere verilir.
"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.NaN, strategy="mean")

yas = veriler.iloc[:,1:4].values
print(yas)

imputer = imputer.fit(yas[:,1:4]) #eğitim
yas[:,1:4] = imputer.transform(yas[:,1:4]) #uygulama
print(yas)