# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:07:06 2024

@author: esraablak
"""

"""
Adım 1: her aksiyon için aşağıdaki iki sayıyı hesaplayınız
    Ni1(n): o ana kadar ödül olarak 1 gelmesi sayısı
    Ni0(n): o ana kadar ödül olarak 0 gelmesi sayısı
    
Adım 2: Her ilan için Beta dağılımında bir rastgele sayı üretiyoruz
Adım 3: En yüksek beta değerine sahip olanı seçiyoruz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv(r"C:\Users\ESRA  ABLAK\Desktop\MachineLearning\Reinforcement\UCB\Ads_CTR_Optimisation.csv")

import random

N = 10000
d = 10
total = 0
selected = []
zeros = [0] * d
ones = [0] * d

for n in range(1,N):
    ad = 0
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate(ones[i] + 1, zeros[i] + 1)
        if (rasbeta > max_th):
            max_th = rasbeta
            ad = i
        selected.append(ad)
        prize = veriler.values[n,ad]
        if (prize == 1):
            ones[ad] = ones[ad] + 1
        else:
            zeros[ad] = zeros[ad] + 1
        total = total + prize
            
    
print("Toplam ödül ", total)    
plt.hist(selected)
plt.show()    
    