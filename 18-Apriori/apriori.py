# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:48:56 2022

@author: reco1
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("sepet.csv", header = None)

t = []
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])

from apyori import apriori
kurallar = apriori(t,min_support = 0.01, min_confidence = 0.2, min_lift = 3,  min_len = 2)              

#ürünlerin birlikte alınma ihtimalinin 3.2 misli arttığını gösterir
print(list(kurallar))


