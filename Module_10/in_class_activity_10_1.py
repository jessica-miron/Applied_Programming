##########################################################
## BME 210:  matplotlib_sklearn_linreg_dino.py          ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed by: Plaisier Lab                          ##
##   (https://plaisierlab.engineering.asu.edu/)         ##
##   Arizona State University                           ##
##   242 ISTB1, 550 E Orange St                         ##
##   Tempe, AZ  85281                                   ##
## @Author:  Chris Plaisier                             ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################

## 1. Load up the packages you will need to run
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


## Load up data
extant_animals = pd.read_csv('animal_species.csv', header=0, index_col=3, encoding='latin-1')


## PdfPages
with PdfPages('scatterPlots_circ_vs_body_mass.pdf') as savedPDF:
    # Pairplot
    sns.pairplot(extant_animals)
    savedPDF.savefig()
    plt.close()
    
    # Pairplot of log transformed data
    sns.pairplot(np.log10(extant_animals.iloc[:,-5:]))
    savedPDF.savefig()
    plt.close()
    
    # Humerus circumference vs. Body Mass grams (log scale)
    plt.scatter(extant_animals.loc[:,'Body_Mass_grams'],extant_animals.loc[:,'Humerus_Circumference'])
    plt.ylabel('Body Mass grams')
    plt.xlabel('Humerus Circumference')
    plt.xscale('log')
    plt.yscale('log')
    savedPDF.savefig()
    plt.close()
    
    # Femur circumference vs. Body Mass grams (log scale)
    plt.scatter(extant_animals.loc[:,'Body_Mass_grams'],extant_animals.loc[:,'Femur_Circumference'])
    plt.ylabel('Body Mass grams')
    plt.xlabel('Femur Circumference')
    plt.xscale('log')
    plt.yscale('log')
    savedPDF.savefig()
    plt.close()


## Linear regression

# Prepare the data
# Already have a pd.DataFrame! Might need to log?

# Build the model
model1 = smf.ols('Body_Mass_grams ~ Femur_Circumference', data = extant_animals)

# Fit the model
results1 = model1.fit()

# Collect results
print(results1.summary())


## Logging the data

# Build the model
model2 = smf.ols('np.log10(Body_Mass_grams) ~ np.log10(Femur_Circumference)', data = extant_animals)

# Fit the model
results2 = model2.fit()

# Collect results
print(results2.summary())


## Plot with regression line
with PdfPages('bmg_femur_circ_model.pdf') as savedPDF:
    pred_ols = results2.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.log10(extant_animals.loc[:,'Femur_Circumference'])
    y = np.log10(extant_animals.loc[:,'Body_Mass_grams'])
    ax.plot(x, y, "o", label="data")
    ax.plot(x, results2.fittedvalues, "r--.", label="OLS")
    ax.plot(x, iv_u, "r--")
    ax.plot(x, iv_l, "r--")
    plt.ylabel('log10(Body_Mass_grams)')
    plt.xlabel('log10(Femur_Circumference)')
    ax.legend(loc="best")
    savedPDF.savefig()
    plt.close()


## Predict for a dinosaur Bambiraptor feinbergi
print(results2.predict(pd.DataFrame({'Bambiraptor feinbergi': {'Femur_Circumference':31.9}}).T))
print((10**3.8)/1000)
