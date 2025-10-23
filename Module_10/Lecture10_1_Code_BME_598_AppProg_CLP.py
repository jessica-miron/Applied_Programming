#############################################################
## BME 598:  in_class_activity_9_2.py                      ##
##  ______     ______     __  __                           ##
## /\  __ \   /\  ___\   /\ \/\ \                          ##
## \ \  __ \  \ \___  \  \ \ \_\ \                         ##
##  \ \_\ \_\  \/\_____\  \ \_____\                        ##
##   \/_/\/_/   \/_____/   \/_____/                        ##
## @Developed by: Plaisier Lab                             ##
##   (https://plaisierlab.engineering.asu.edu/)            ##
##   Arizona State University                              ##
##   242 ISTB1, 550 E Orange St                            ##
##   Tempe, AZ  85281                                      ##
## @Author:  Chris Plaisier                                ##
## @License:  GNU GPLv3                                    ##
##                                                         ##
## If this program is used in your analysis please         ##
## mention who built it. Thanks. :-)                       ##
#############################################################

## 1. Load up the packages you will need to run
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(9876789)


## 2. Simulate some data
# 100 samples
nsample = 100

# Makes a linear increasing float np.ndarray starting at 0 to 10 with 100 steps in between
x = np.linspace(0, 10, 100)

# Makes a two column np.ndarray with x in one column and x squared in the next column
X = np.column_stack((x, x ** 2))

# Add another column that is just 1
X = sm.add_constant(X)

# Create the errors using a normal distribution
e = np.random.normal(size=nsample)

# Define beta values for each column in X
beta = np.array([1, 0.1, 10])

# Compute y using a linear model
y = np.dot(X, beta) + e


## 3. Ordinary least squares (OLS) modeling

# Build model
model = sm.OLS(y, X)

# Fit model
results = model.fit()

# Collect results
print(results.summary())


## 4. Alternative OLS method using a formula

# Turn X into a pd.DataFrame
df1 = pd.DataFrame(X, columns=['C','x1','x2'])
df1['y'] = y

# Build model
model2 = smf.ols('y ~ x1 + x2', data=df1)

# Fit model
results2 = model2.fit()

# Collect results
print(results2.summary())
