#############################################################
## BME 210:  seaborn_ttest_boxplot_violinplot_swarmplot.py ##
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

## Import
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import ttest_ind
#import numpy as np

## Load up data
# CEMETERY - where the bones were found
# SEX - sex if known/determined of remains
# AGE - age range for samples
# VALUE - femure length
bones = pd.read_csv('bones_London_medieval_postMedieval_FeL_age.csv', header=0, index_col=1)

## Explore the data (text based)
# Info first
bones.info() # Some missing data for AGE

# Values for SEX
bones.loc[:,'SEX'].value_counts()

# Values for AGE
bones.loc[:,'AGE'].value_counts()

# Values for CEMETERY
bones.loc[:,'CEMETERY'].value_counts()


## Convert to stature
bones.loc[:,'stature'] = (2.38*(bones.loc[:,'VALUE']/10))+61.41


## Histogram of stature
with PdfPages('histogram_stature.pdf') as pdf:
    # In matplotlib all samples
    plt.hist(bones.loc[:,'stature'])
    plt.ylabel('Probability')
    plt.xlabel('stature (cm)')
    pdf.savefig()
    plt.close()
    # Seaborn histogram
    sns.histplot(bones, x='stature')
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()

# Histogram of stature stratified by SEX
with PdfPages('histogram_stature_by_SEX.pdf') as pdf:
    # Seaborn histogram
    sns.histplot(bones, x='stature', hue='SEX')
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()

# Histogram of stature stratified by AGE
with PdfPages('histogram_stature_by_AGE.pdf') as pdf:
    # Seaborn histogram
    sns.histplot(bones, x='stature', hue='AGE')
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()


## Boxplots
with PdfPages('boxplot_stature.pdf') as pdf:
    # Seaborn boxplot
    sns.boxplot(x='SEX', y='stature', data=bones)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    sns.boxplot(x='AGE', y='stature', data=bones)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()

## Violinplots
with PdfPages('violinplot_stature.pdf') as pdf:
    # Seaborn boxplot
    sns.violinplot(x='SEX', y='stature', data=bones)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    sns.violinplot(x='AGE', y='stature', data=bones)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()

## Swarmplots
with PdfPages('swarmplot_stature.pdf') as pdf:
    # Seaborn boxplot
    sns.swarmplot(x='SEX', y='stature', data=bones)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    sns.swarmplot(x='AGE', y='stature', data=bones)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()


## Box+Swarmplots
with PdfPages('boxplot_swarmplot_stature.pdf') as pdf:
    # Seaborn boxplot
    ax = sns.boxplot(x='SEX', y='stature', data=bones)
    ax = sns.swarmplot(x='SEX', y='stature', data=bones, edgecolor='white', linewidth=0.5)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    ax = sns.boxplot(x='SEX', y='stature', data=bones)
    ax = sns.swarmplot(x='SEX', y='stature', data=bones, edgecolor='white', linewidth=0.5)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()



## T-test
ttest_ind(bones.loc[bones.loc[:,'SEX']=='MALE','stature'], bones.loc[bones.loc[:,'SEX']=='FEMALE','stature'])

ttest_ind(bones.loc[bones.loc[:,'SEX']=='FEMALE','stature'], bones.loc[bones.loc[:,'SEX']=='MALE','stature'])

