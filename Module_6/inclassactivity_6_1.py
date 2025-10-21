# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:05:03 2025

@author: jcmir
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

## Load up data
# CEMETERY - where the bones were found
# SEX - sex if known/determined of remains
# AGE - age range for samples
# VALUE - femure length
bones = pd.read_csv('bones_London_medieval_postMedieval_FeL_age.csv',header=0, index_col=1)

## Convert to stature
bones.loc[:,'stature'] = (2.38 * (bones.loc[:,'VALUE']/10)) + 61.41

# Seaborn histogram
sns.histplot(bones, x='stature')
plt.tight_layout()
plt.show()
#pdf.savefig()
plt.close()

# Histogram of stature stratified by SEX
with PdfPages('histogram_stature_by_SEX.pdf') as pdf :
# Seaborn histogram
    sns.histplot(bones, x='stature', hue='SEX')
    pdf.savefig()
    plt.show()
    plt.close()
# Histogram of stature stratified by AGE
with PdfPages('histogram_stature_by_AGE.pdf') as pdf:
# Seaborn histogram
    sns.histplot(bones, x='stature', hue='AGE')
    pdf.savefig()
    plt.show()
    plt.close()
    
with PdfPages('boxplot_stature.pdf') as pdf:
# Seaborn boxplot
    sns.boxplot(x='SEX', y='stature', data=bones)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()
    
## Swarmplots overlaid on boxplots
with PdfPages('swarmplot_stature.pdf') as pdf:
# Seaborn swarmplot
    ax = sns.boxplot(x='SEX', y='stature', data=bones)
    ax = sns.swarmplot(x='SEX', y='stature', data=bones, edgecolor='white',
                       linewidth=0.5)
    plt.xticks(rotation=30)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    plt.close()


