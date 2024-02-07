# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:09:15 2024

@author: esraablak
"""

"""
VERİ
    I) Kategorik
        - Nominal : Ne sıralanabilen ne ölçülebilen. Araba markaları, telefon markaları... Kendi içerisinde 2'ye ayrılır:
                a) Binomial (kadın,erkek vs.)
                b) Polinomial (tr,abd,fr,sp vs.)
        - Ordinal : Order kelimesinden gelir. Sıraya sokulabilen, aralarında büyüktür küçüktür ilişkisine girebilen ama ölçülemeyen şeyler. Plakalar sıralanabilir (34,06..) ama bu sıra herhangi bir ölçüm belirtmez. anket memnuniyet seviyesi (çok memnun, memnun, hiç memnun değil vs)
    II) Sayısal
        - Oransal (Ratio) : Birbirlerine göre orantılanabilen, çarpılıp bölünebilen değerler. Örn yaş.
        - Aralık (Interval) : Herhangi bir çarpma gibi işlemleri kabul etmeyen, belirli bir aralıkta olan değerler. Örn sıcaklık ölçümü. Değerler için büyüktür küçüktür ilişkisi var, değerin artması veya azalması mümkün. Ama çarpamnın bi anlamı yok.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")

ulke = veriler.iloc[:,0:1]
print(ulke)

from sklearn import preprocessing

"""
LabelEncoding: Kategorik veerilerin sırasıyla artan etiketlere dönüştürülmesini sağlar.
Örn Kırmızı->0 Yeşil->1 Mavi->2
OneHotEncoding: Her bir kategorik değeri bağımsız bir sütun olarak ele alır ve her bir sütunu 0 veya 1 olarak işaretler. Örn "Kırmızı" -> [1, 0, 0], "Yeşil" -> [0, 1, 0], "Mavi" -> [0, 0, 1] 
"""

#le = preprocessing.LabelEncoder()
#ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
#print(ulke)

"******************************************************************************"

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)