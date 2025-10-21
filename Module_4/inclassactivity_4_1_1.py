# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:28:04 2025

@author: jcmir
"""

import pandas as pd
df1 = pd.DataFrame([['pizza','jolt_cola'],['beans','tea'],['hamburger','coca_cola']],index =
['Bill', 'Joe', 'Umu'], columns = ['Food', 'Drink'])

df1.to_csv('lunch_DataFrame.csv')

panCancer_phenos = pd.read_csv('phenotypes_panCancer.csv', header = 0, index_col=0)
print(panCancer_phenos.head)
print(panCancer_phenos.tail)
print(panCancer_phenos.shape)
print(panCancer_phenos.dtypes)
print(panCancer_phenos.info())
print(panCancer_phenos.describe())
print(panCancer_phenos.loc[:, 'tumor'].value_counts())
print(panCancer_phenos['tumor'].value_counts().shape)

panCancer_phenos_stad = panCancer_phenos.loc[ panCancer_phenos['tumor'] == 'STAD',: ]
print(panCancer_phenos_stad)
panCancer_phenos_stad_yt40 = panCancer_phenos.loc[(panCancer_phenos['tumor'] == 'STAD') 
                                                  &(panCancer_phenos['age_at_initial_pathologic_diagnosis'] <= 40), :]
print(panCancer_phenos_stad_yt40)
panCancer_phenos.loc[ ~(panCancer_phenos.tumor == 'STAD'),:]

panCancer_GBM = panCancer_phenos.loc[panCancer_phenos['tumor']=='GBM',:]
print(panCancer_GBM.shape)
print(panCancer_phenos['gender'])

panCancer_GBM_Male = panCancer_phenos.loc[(panCancer_phenos['tumor']=='GBM') & (panCancer_phenos['gender']=='MALE'),:]
print(panCancer_GBM_Male.shape)
