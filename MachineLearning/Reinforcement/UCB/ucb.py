# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:56:48 2024

@author: esraablak
"""

"""
UCB (Upper Confidence Bound)
    üst güven sınırı: her olayın arkasında bir dağılım var.
    dağılımları nasıl "en avantajlı hale çeviririz?"
        * Kullanıcı her seferinde bir eylem yapar
        * Bu eylem karşılığında bir skor döner (örneğin web tıklanması 1 ve tıklanmaması 0)
        * Amaç tıklanmaları maksimuma çıkarmak
        
        
Adım 1: Her turda (tur sayısı n olsun), her reklam alternatifi (i için) aşağıdaki sayılar tutulur
    Ni(n): i sayılı reklamın o ana kadarki tıklanma sayısı
    Ri(n): o ana kadar ki reklamdan gelen toplam ödül
Adım 2: Yukarıdaki bu iki sayıdan, aşağıdaki değerler hesaplanır:
    O ana kadarki her reklamın ortalama ödülü Ri(n) / Ni(n) => ucb
    Güven aralığı için aşağı ve yukarı oynama potansiyeli => delta
Adım 3: En yüksek UCB değerine sahip olanı alırız
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

"***********************************************************************************************************************"

import math

N = 10000
d = 10
prizes = [0] * d # Ri(n)
clicks = [0] * d # Ni(n)
total = 0
selected = []
for n in range(0,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if (clicks[i] > 0):
            mean = prizes[i] / clicks[i]
            delta = math.sqrt(3/2 * math.log(n)/clicks[i])
            ucb = mean + delta
        else:
            ucb = N * 10
        if (max_ucb < ucb):
            max_ucb = ucb
            ad = i
            
    selected.append(ad)
    clicks[ad] = clicks[ad] + 1
    prize = veriler.values[n,ad]
    prizes[ad] = prizes[ad] + prize
    total = total + prize
    
    
print("Toplam ödül ", total)    
plt.hist(selected)
plt.show()    
    
    
    
    
    
    
    
    
    