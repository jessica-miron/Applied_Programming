# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:04:10 2025

@author: jcmir
"""

# Jessica Miron 1224633969

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

fev = pd.read_table("fev_dat.txt",header=0)
print(f"fev shape: {fev.shape}")

with PdfPages('fev_pairplots.pdf') as savedPDF:
    # Pairplot
    sns.pairplot(fev)
    savedPDF.savefig()
    plt.close()
    # Age, FEV, and height all positive linear relationships to each other
    # Smoke and sex categorical
print("Pairplot created")

print("Model 1: FEV vs Smoking")
model1 = smf.ols('FEV ~ smoke',data=fev)
results1 = model1.fit()
print(results1.summary())
# Positive 0.7 of smoking says smoking has positive impact on FEV but model is 
# not very predictive because of R^2 value
# All terms are significant because p<0.05

print("Model 2: FEV vs smoking and age")
model2 = smf.ols('FEV ~ age + smoke',data=fev)
results2 = model2.fit()
print(results2.summary())
# Positive 0.2 means FEV increases 0.2 liters/second with 1 year age and 
# -0.2 means FEV decreases 0.2 liters/second if smoking. 
# Model is moderately predictive with R^2 of 0.577
# All terms are significant because p<0.05, though smoking is less significant 
# then age and the constant

print("Model 3: FEV vs quadratic height")
#model3=smf.ols('FEV ~ age + smoke + ht + np.power(ht,2), data=fev')
model3=smf.ols('FEV ~ age + sex + ht + np.power(ht,2) + smoke',data=fev)
results3=model3.fit()
print(results3.summary())
# Best performing model so far with R2 0.790
# All terms are significant because p < 0.05
# Age increases FEV 0.06 liters/second per year
# Smoking decreases FEV 0.15 liters/second 
# Height decreases FEV 0.30 liters/second per inch
# Height squared increases FEV 0.0034 liters/second per inch squared
# Without quadratic height smoking is no longer significant and has 0.75 ishR2

# Want to minimize AIC and BIC to determine best model currently best is quad height
# better than model with just height because AIC and BIC are lower

print("Model 4: interactions")
#model4=smf.ols('FEV ~ age*smoke',data=fev)
model4=smf.ols('FEV ~ age + smoke + age:smoke',data=fev)
results4 = model4.fit()
print(results4.summary())
# All terms significant
# Age increases FEV 0.24
# Smoking greatly increases FEV 1.94
# interaction decreases FEV -0.16
# R2 0.592
# AIC BIC have gone back up a lot but is better than just smoking and age

print("Model 5: everything")
#model5 = smf.ols('FEV ~ age*smoke + ht + np.power(ht,2)',data=fev)
model5=smf.ols('FEV ~ age + sex + ht + np.power(ht,2) + smoke + age:smoke',data=fev)
results5=model5.fit()
print(results5.summary())
# AIC and BIC have increased so model is less good and R2 is the same
# Now smoke and age smoke interaction no longer significant
# Age increases 0.07, Smoke increases 0.16, age:smoke decreases 0.02
# height decreases 0.31, quad height increases 0.0034

print("Model 6: custom")
model6=smf.ols('FEV ~ age*ht*sex + smoke', data=fev)
results6=model6.fit()
print(results6.summary())