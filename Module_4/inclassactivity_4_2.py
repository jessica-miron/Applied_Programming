# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 08:59:57 2025

@author: jcmir
"""
import pandas as pd

panCancer_phenos = pd.read_csv('phenotypes_panCancer.csv', header = 0, index_col=0)
print(panCancer_phenos.head())
print(panCancer_phenos.columns)
print(panCancer_phenos.isnull().sum())
panCancer_phenos_STAD = panCancer_phenos.loc[panCancer_phenos['tumor']=='STAD']
print('before NA')
print(panCancer_phenos_STAD.shape)
print('\n')
panCancer_phenos_STAD_noNA = panCancer_phenos_STAD.dropna(
    subset=['OS','OS.time','age_at_initial_pathologic_diagnosis','gender'])
print('after NA')
print(panCancer_phenos_STAD_noNA.shape)
print('\n')
panCancer_phenos_STAD_allNA = panCancer_phenos_STAD.dropna()
print('all NA gone')
print(panCancer_phenos_STAD_allNA.shape)

leukocyte_TCGA = pd.read_csv('leukocyte_TCGA.csv',header=0, index_col=0)
cancer = pd.merge(panCancer_phenos,leukocyte_TCGA,on='bcr_patient_barcode')

cancer_LUSC= cancer.loc[cancer['tumor']=='LUSC']
print(cancer_LUSC['TotalLeukocyte'].median())
cancer_ACC= cancer.loc[cancer['tumor']=='ACC']
print(cancer_ACC['TotalLeukocyte'].median())
cancer_COAD= cancer.loc[cancer['tumor']=='COAD']
print(cancer_COAD['TotalLeukocyte'].median())
cancer_UVM= cancer.loc[cancer['tumor']=='UVM']
print(cancer_UVM['TotalLeukocyte'].median())

groups=cancer.groupby('tumor')
print(groups['TotalLeukocyte'].median())

cancer_noNA = cancer.dropna(subset=['age_at_initial_pathologic_diagnosis','TotalLeukocyte'])
print(cancer_noNA.shape)

