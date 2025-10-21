# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 09:11:02 2025

@author: jcmir
"""

import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

fev = pd.read_table("fev_dat.txt")

stat, p_value = ttest_ind(fev.loc[fev.loc[:,'sex']==0,'FEV'], fev.loc[fev.loc[:,'sex']==1,'FEV'])
print('Sex vs FEV')
print(stat,p_value)
stat, p_value = ttest_ind(fev.loc[fev.loc[:,'age']<10,'FEV'], fev.loc[fev.loc[:,'age']>=10,'FEV'])
print('Age vs FEV')
print(stat,p_value)
stat, p_value = ttest_ind(fev.loc[fev.loc[:,'smoke']==0,'FEV'], fev.loc[fev.loc[:,'smoke']==1,'FEV'])
print('Smoker vs FEV')
print(stat,p_value)

ax = sns.scatterplot(x='ht',y='FEV',hue='age',data=fev)
plt.xlabel('Height (in)')
plt.ylabel('FEV (L/s)')
plt.tight_layout()
plt.legend(title='Age (years)')
plt.show()
plt.close()