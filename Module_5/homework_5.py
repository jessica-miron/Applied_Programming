# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 10:32:53 2025

@author: jcmir
"""

# Jessica Miron 1224633969

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Part 1
panCancer_phenos = pd.read_csv('phenotypes_panCancer.csv', header = 0, index_col=0)
print("First 5 Rows")
print(panCancer_phenos.head())
print("Column Names")
print(panCancer_phenos.columns)
print("Info")
print(panCancer_phenos.info())
print("Description")
print(panCancer_phenos.describe())

panCancer_phenos_noNA = panCancer_phenos.dropna(
    subset=['tumor','age_at_initial_pathologic_diagnosis','gender','OS.time'])
print('Info Minus Nulls')
print(panCancer_phenos_noNA.info())

with PdfPages('Homework_5_Plots.pdf') as pdf:
    # Part 2
    tumor_types = ['BLCA','LUAD','SKCM']
    three_cancer = panCancer_phenos_noNA[panCancer_phenos_noNA['tumor'].isin(tumor_types)]

    age_at_diagnosis = [three_cancer[three_cancer['tumor']==t]['age_at_initial_pathologic_diagnosis']
        for t in tumor_types]

    plt.figure(figsize=(6,5),layout='constrained')
    plt.violinplot(age_at_diagnosis, showmeans=True)
    plt.xticks(np.arange(1,len(tumor_types)+1), tumor_types)
    plt.xlabel("Tumor Type")
    plt.ylabel("Age at Initial Pathologic Diagnosis (years)")
    plt.title("Age at Initial Pathologic Diangosis by Tumor Type")
    pdf.savefig()
    plt.show()
    plt.close()
    
    # Part 3
    genders = ['MALE','FEMALE']
    survival_time = [three_cancer[three_cancer['gender']==g]['OS.time']
        for g in genders]
    plt.figure(figsize=(6,5),layout='constrained')
    plt.violinplot(survival_time, showmeans=True)
    plt.xticks(np.arange(1,len(genders)+1), genders)
    plt.xlabel("Gender")
    plt.ylabel("Overall Survival Time (days)")
    plt.title("Overall Survival Time by Gender for BLCA, LUAD, SKCM")
    pdf.savefig()
    plt.show()
    plt.close()
    
    # Part 4 DONT TAKE OUT OTHERS AND ROTATE LABELS INCLASS 6_1
    lung_adenocarcinoma = panCancer_phenos_noNA.loc[panCancer_phenos_noNA['tumor']=='LUAD']
    races = lung_adenocarcinoma['race'].unique()
    race_age_at_diagnosis = [lung_adenocarcinoma[lung_adenocarcinoma['race']==r]['age_at_initial_pathologic_diagnosis']
                             for r in races]
    plt.figure(figsize=(12,5),layout='constrained')
    plt.violinplot(race_age_at_diagnosis,showmeans=True)
    plt.xticks(np.arange(1,len(races)+1),races)
    plt.xticks(rotation=30)
    plt.xlabel("Race")
    plt.ylabel("Age at Initial Pathologic Diagnosis (years)")
    plt.title("Age at Initial Pathologic Diagnosis by Race for LUAD")
    pdf.savefig()
    plt.show()
    plt.close()
    
