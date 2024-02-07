# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:11:06 2024

@author: esraablak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

import random

N = 10000
d = 10
total = 0
selected = []
for n in range(0,N):
    ad = random.randrange(d)
    selected.append(ad)
    prize = veriler.values[n,ad]
    total = total + prize
    
plt.hist(selected)
plt.show()