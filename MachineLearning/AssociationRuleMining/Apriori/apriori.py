# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:59:21 2024

@author: esraablak
"""

"""
ctrl+I = help

apriori (sebep-sonuç ilişkisi)
bunu alanlar bunu da aldı, bunu izleyenler bunu da izlerdi..

Support(a) = a varlığını içeren eylemler / toplam eylem sayısı
Confidence(a->b) = a ve b varlığını içeren eylemler / a varlığını içeren eylemler
Lift(a->b) = Confidence(a->b) / Support(b)
    lift değeri: a eyleminin b eylemine etkisi
        lift > 1 = a ürününü alanların b ürününü alma ihtimalini arttırıyor, olumlu arttırıyor.
        lift < 1 = olumsuz etkiliyor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("sepet.csv", header=None)

t = []

for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])


from apyori import apriori

rules = apriori(t, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
rules_list = (list(rules))

for item in rules_list:
 
    base_items = [x for x in item[2][0][0]]
    add_item, = item[2][0][1]
    print("Rule: " + " + ".join(base_items) + " -> " + str(add_item))
 
    print("Support: " + str(item[1]))
 
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")