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
import pandas as pd
import GEOparse
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm

import mygene

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

import gzip
import pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
#%%
## 2. Load up and inspect data and function

# Use GEOparse to get data
gse_id = "GSE179882"

stem_cell_data = GEOparse.get_GEO(geo= gse_id, annotate_gpl=(True))
print("GEO dataset downloaded: "+gse_id)

print("\nMetadata:")
print(stem_cell_data.metadata.keys())
print(stem_cell_data.metadata["title"])
print(stem_cell_data.metadata["type"])
#%%
# List the samples in this dataset
num_samples = len(stem_cell_data.gsms)
print("Number of samples: "+str(num_samples))
print(list(stem_cell_data.gsms.keys())[:num_samples])
#%%
# Look at the data stored for each GSM
stem_cell_data.gsms['GSM5436464'].columns
#%%
# Since this is an RNA sequencing data set, we are expecting a table of gene counts

print(stem_cell_data.metadata) # look to see what is found in this GEO entry
#  see that there is a 'supplmentary_file' that seems like counts
#  the supplementary file is a URL, so need to have internet signal to read it
#%%
# Load the data using URL
expression_matrix = pd.read_csv(stem_cell_data.metadata['supplementary_file'][0], compression='gzip', low_memory=False, header=0, index_col=0)

# Fixing the name of the index to be spellec correctly
expression_matrix.index.name = 'Accession'

# Check out the data
print("\nData preview:")
print(expression_matrix.head())

# Remove Title
expression_matrix = expression_matrix.drop('Title')

# Check that this all worked
print(expression_matrix.head())

# Save to file
expression_matrix.to_csv(gse_id+"_expression_matrix.csv", index=False)
#%%
# Make annotation table that maps Ensembl IDs to gene symbols
print("\nMapping Ensembl IDs to gene symbols:")
mg = mygene.MyGeneInfo()
ensembl_ids = expression_matrix.index.tolist() # all Ensembl IDs in dataset
ensembl_ids_no_version = [x.split('.')[0] for x in ensembl_ids] # strip version
annot = mg.querymany(ensembl_ids_no_version, scopes='ensembl.gene', fields='symbol', species='human', as_dataframe=True)
annot = pd.DataFrame(annot)
annot = annot.rename(columns={'query': 'Accession'})

## Small clean up for expression_matrix
# First convert to integers
expr_data = expression_matrix.astype(int)

# Then get rid of rows that don't have at least 10 counts total
expr_data = expr_data[expr_data.sum(axis=1) > 10]

# Then transpose
expr_data = expr_data.T

## Build metadata
# Get the sample descriptions so we can set the samples between which we want to get differentially expressed genes
sample_info = stem_cell_data.phenotype_data

# Convert to a DataFrame
print("\nSample descriptions:")
print(sample_info["title"]) # take a look
#  see that there are several lines with shRNA to HDAC1 or NT (non-targetting control)
#%%
# 3. Setup metadata
sample_info['group'] = [i.split(' ')[0]+'_'+i.split(' ')[1] for i in sample_info['title']]

# List of lists that contain labels to mark each group in the sample info table
group_labels = [['BT145_shNT','BT145_shHDAC1'],['BT187_shNT','BT187_shHDAC1'],['CC24_shNT','CC24_shHDAC1'],['GB3_shNT','GB3_shHDAC1'],['NHA_shNT','NHA_shHDAC1']]

# Save to sample info to CSV
sample_info.to_csv(gse_id+"_sample_metadata.csv", index=False)
#  sample info file will get overwritten as more groups will be filled out


## Create DESeqDataSet object
dds = DeseqDataSet(counts = expr_data,
                   metadata = sample_info,
                   design="~group")

## Perform normalization, dispersion estimation, and logFC fitting
dds.deseq2()


## Iterate through the comparisons, labels for which have already been loaded into sample_info
for i in range(len(group_labels)):
    
    # Grab labels for current comparison
    label = group_labels[i]
    group1_label = label[0]
    group2_label = label[1]     

    # Use pydeseq2 to analyze differential expression
    print("\nCalculating differential expression:")
    print("\nGroup 1 = "+group1_label+", Group 2 = "+group2_label)
    
    # Make a DeseqStats object and define a contrast that will comapre the shHDAC1 vs. shNT
    ds = DeseqStats(dds, contrast=['group', group2_label, group1_label])
    
    # Collect differential expression calculations
    ds.summary()
    
    # Extract results pd.DataFrame
    res_df = ds.results_df
    
    # Sort the genes by significance
    res_df = res_df.sort_values('padj')

    # Redo the index so starts with 0 and goes up
    res_df = res_df.reset_index()

    print("\nPreviewing differential expression results:")
    print(res_df.head())
    # Note: log2FC is called 'log2FoldChange' and FDR is called 'padj'
    print(res_df.columns)
    
    ## Add gene symbol to results 
    # Add a column for Ensembl IDs without the version 
    res_df["Ensembl_ID_no_version"] = [x.split('.')[0] for x in res_df["Accession"]]
    print("\nPreviewing differential expression results with Ensembl IDs:")
    print(res_df.head())
    print(res_df.columns) 
    
    # Merge with the annotation table made previously
    res_df = res_df.merge(
        annot[['Accession', 'symbol']],
        how='left',
        left_on='Ensembl_ID_no_version',
        right_on='Accession'
    )
    res_df = res_df.drop(columns=['Accession_y']) # redundant
    res_df = res_df.drop(columns=['Ensembl_ID_no_version']) # don't need any more
    res_df = res_df.rename(columns={'Accession_x': 'Accession'}) # rename for clarity
    print("\nPreviewing differential expression results with gene symbols:")
    print(res_df.head())
    print(res_df.columns) 
    
    ## Export differential expression results to a file
    res_df.to_csv(gse_id+"_DiffExpr_"+group1_label+"_vs_"+group2_label+".csv", index=False)
    # Note that looking through these results, all genes will have a 'baseMean' value,
    #     but other columns are blank if the value can not be calculated


    ## Volcano Plot
    
    log2FC_threshold = 1
    pval_threshold = 0.05
    
    # Volcano plot with up- and down-regulated genes labeled
    
    # Grab genes that are up or down using the thresholds
    down = res_df[(res_df['log2FoldChange'] <= -1 * log2FC_threshold) & (res_df['padj'] <= pval_threshold)]
    up = res_df[(res_df['log2FoldChange'] >= log2FC_threshold) & (res_df['padj'] <= pval_threshold)]
    
    # Same scatter plot as before with up- and down- regulated genes overlaid 
    #   in different colors
    
    print("\nUsing log2FC threshold of "+str(log2FC_threshold)+" and FDR p-value threshold of "+str(pval_threshold)+":")
    print("Number of signficant upregulated genes: "+str(up.shape[0]))
    print("Number of signficant downregulated genes: "+str(down.shape[0]))
    
    with PdfPages(gse_id+"_VolcanoUpDown_"+group1_label+"_vs_"+group2_label+".pdf") as pdf:
        plt.figure(figsize=(6, 6))
        plt.scatter(x=res_df['log2FoldChange'],y=res_df['padj'].apply(lambda x:-np.log10(x)),s=1, color = "lightgrey") 
        plt.scatter(x=up['log2FoldChange'],y=up['padj'].apply(lambda x:-np.log10(x)),s=3,label="Up-regulated",color="red")
        plt.scatter(x=down['log2FoldChange'],y=down['padj'].apply(lambda x:-np.log10(x)),s=3,label="Down-regulated",color="green")
        plt.xlabel("log2FC")
        plt.ylabel("-log10 FDR")
        plt.axvline(-1*log2FC_threshold,color="grey",linestyle="--")
        plt.axvline(log2FC_threshold,color="grey",linestyle="--")
        plt.axhline(-np.log10(pval_threshold),color="grey",linestyle="--")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
