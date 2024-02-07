# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 08:24:15 2024

@author: esraablak
"""

"""
- Kaç küme olucağı kullanıcıdan parametre olarak seçilir.
- Rasgele olarak k merkez noktası seçilir.
- Her veri örneği en yakın merkez noktasına göre ilgili kümeye atanır.
- Her küme için yeni merkez noktaları hesaplanarak merkez noktaları kaydırılır.
- Yeni merkez noktalarına göre 


K-Means için kümelemenin en iyi şekilde yapıldığını nasıl anlarsın? Küme içerisindeki verilerin ağırlık merkezine olan mesafesinin minimumda ve kümenin diğer kümelere olan mesafesinin maksimumda olmasıyla. K-Means'da k değerleri rastgele alındığı için bu problemle karşılaşılabiliyor. yani evet kümeleme yapıyor ama en iyi şekilde yapmamış olabilir. Bunun önüne geçilmek için K-Means++ algortiması geliştirilmiş.
    1. Rasgele seçilen noktalardan en yakınına her noktadan uzaklığı hesaplama (buna D(x) = Distance(x) diyelim)
    2. Yeni noktaları mesafenin karesini olasılık alarak (D(x)^2 ile bul)
    
k değerine nasıl karar verirsin?
WCSS (within-cluster sums of squares): k değeri birden başlayarak wcss hesaplarını buluyo her bi wcss değerini bir grafiğe alıyor. wcss değerinin az olması ağırlık merkezine olan mesafe az demektir. grafiğe bakıldığında wcss değerine göre en optimum k değerini sen seçersin. k değeri arttıkça wcss azalır ama bu wcss'nin 0 değerini bulmasına kadar gider yani her bir veri için bi kümeleme yapar. bu en iyisi demek değil. wcss azalma boyutlarına göre optimum değerini sen seçersin.

x-means => bu algoritmada x'i makine kendisi buluyor. örn 2-50 arasında dene diyosun. tek tek deniyo optimumu kendisi buluyor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=1)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) # wcss değerlerini ekler


plt.plot(range(1,11),sonuclar)