#############################################################
## BME 598:  in_class_activity_9_1.py                      ##
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

##################################################
## 1. Load up the packages you will need to run ##
##################################################

# Add data loading and manipulation
import pandas as pd
import GEOparse
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind

# Add statistics
import statsmodels.stats.multitest as smm
from scipy.stats import hypergeom

# Add plotting
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

# Add status bar as the differential expression analysis is running
from tqdm import tqdm


##############################################
## 2. Load up and inspect data and function ##
##############################################

# Use GEOparse to get data
gse_id = 'GSE17300'

# Could modify where the data is stored by setting destdir
usf1_data = GEOparse.get_GEO(geo= gse_id, annotate_gpl=(True))
print('GEO dataset downloaded: '+gse_id)

# What is in the metadata?
print('\nMetadata:')
print(usf1_data.metadata.keys())
print(usf1_data.metadata['title'])
print(usf1_data.metadata['type'])

# List the samples in this dataset
num_samples = len(usf1_data.gsms)
print('Number of samples: '+str(num_samples))
print(list(usf1_data.gsms.keys())[:num_samples])

# Extract the expression data from each sample and merge into one matrix
#   Each GSM has a .table with columns (commonly 'ID_REF' and 'VALUE')
#   Loop through each sample extracting the table and append together

# Load expression matrix
expression_matrix = usf1_data.pivot_samples(values='VALUE', index='ID_REF')

print('\nData preview:')
print(expression_matrix.head())

# Save to file
expression_matrix.to_csv(gse_id+'_expression_matrix.csv')

# Get gene symbols for probe IDs to make it easier to facilitate interpretation of data

# Check which platform(s) this GSE uses
print('\nPlatform:')
print(usf1_data.gpls.keys())
platform_id = list(usf1_data.gpls.keys())[0]

# Load the platform (GPL) annotation table
gpl = usf1_data.gpls[platform_id]

print(gpl.table.columns)
# Common columns: ['ID', 'Gene Symbol', 'Gene Title', 'ENTREZ_GENE_ID', ...]

# Keep only what you need
annot = gpl.table[['ID', 'Gene Symbol','ENTREZ_GENE_ID']].rename(columns={'ID': 'ID_REF'})
# Be sure to look through this as some probe IDs map to more than one symbol (e.g. “ABC1 /// ABC2”)
#   and some do not map to a gene (e.g. NaN)

# Get the sample descriptions so we can set the samples between which we want to get differentially expressed genes
sample_info = usf1_data.phenotype_data

# Convert to a DataFrame
print('\nSample descriptions:')
print(sample_info['title']) # take a look


#####################################
## 3. Differential Gene Expression ##
#####################################

# Define phrases to identify the sample groups
group1_phrase = 'Empty'
group2_phrase = 'USF1'
print('\nPhrases defining groups for differential expression:')
print('Group 1 = '+group1_phrase+'\nGroup 2 = '+group2_phrase)

sample_info['group'] = ''  # initialize empty column

# Fill 'group' column based on which phrase is found in the 'title' column
#   in sample_info
for idx in sample_info.index:
    title = str(sample_info.at[idx, 'title'])

    if group1_phrase in title:
        sample_info.at[idx, 'group'] = 'control'
    elif group2_phrase in title:
        sample_info.at[idx, 'group'] = 'USF1overexp'
    else:
        sample_info.at[idx, 'group'] = 'unknown'  # default value just in case

# Save to sample info to CSV
sample_info.to_csv(gse_id+'_sample_metadata.csv', index=False)

# Split samples (adjust according to data set)
group1 = sample_info.index[sample_info['group'] == 'control']
group2 = sample_info.index[sample_info['group'] == 'USF1overexp']

# t-test for each gene with multiple hypothesis correction
print('\nCalculating differential expression:')
results = []
with tqdm(total = len(expression_matrix.index)) as pbar:
    for gene in expression_matrix.index:
        stat, p = stats.ttest_ind(expression_matrix.loc[gene, group2], expression_matrix.loc[gene, group1])
        median_group1 = expression_matrix.loc[gene, group1].median()
        median_group2 = expression_matrix.loc[gene, group2].median()

        # this data is RMA normalized, which means it is log2 transformed already
        #   so we can just take the difference of the medians as the log foldchange
        foldchange = median_group2 - median_group1

        results.append((gene, stat, median_group1, median_group2, foldchange, p))
        pbar.update(1)

# Create the pd.DataFrame containing the results
res_df = pd.DataFrame(results, columns=['Gene', 't', 'group1_median','group2_median','log2FC', 'pval'])

# Do the multiple hypothesis correction, and add the corrected p-values to a new column FDR
res_df['FDR'] = smm.multipletests(res_df['pval'], method='fdr_bh')[1]

# Sort the data based upon the new FDR column
res_df = res_df.sort_values('FDR')

# Add a column with the gene symbol for each probe ID
# by merging with platform annotation we collected earlier
#   Probe IDs that map to more than one symbol will look like: “ABC1 /// ABC2”
#   Probe IDs that do not map to a gene will be 'nan'

res_annot = res_df.merge(
    annot[['ID_REF', 'Gene Symbol', 'ENTREZ_GENE_ID']],
    how='left',
    left_on='Gene',
    right_on='ID_REF'
)

# Drop the redundant ID_REF column
res_annot = res_annot.drop(columns=['ID_REF'])

print(res_annot.head(10))

# Phrases without spaces to use in output file names
group1_label = 'control'
group2_label = 'USF1overexp'

# Export differential expression results to a file
res_annot.to_csv(gse_id+'_DiffExpr_'+group2_label+'_vs_'+group1_label+'.csv', index=False)


#####################
## 4. Volcano Plot ##
#####################

log2FC_threshold = 1
pval_threshold = 0.1

# Scatter plot of log-transformed foldchange vs -log FDR (multiple hypothesis
#    testing corrected p-value)
with PdfPages(gse_id+'_Volcano_'+group2_label+'_vs_'+group1_label+'.pdf') as pdf:
    plt.figure(figsize=(6, 6))
    plt.scatter(x = res_annot['log2FC'], y = -np.log10(res_annot['FDR']), s=1)
    plt.xlabel('logFC')
    plt.ylabel('-log10 FDR')
    plt.axvline(-1*log2FC_threshold, color = 'grey', linestyle = '--')
    plt.axvline(log2FC_threshold,color='grey',linestyle='--')
    plt.axhline(-np.log10(pval_threshold), color = 'grey', linestyle = '--')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# Volcano plot with up- and down-regulated genes labeled

# Grab genes that are up or down using the thresholds
down = res_annot[(res_annot['log2FC'] <= -1 * log2FC_threshold) & (res_annot['FDR'] <= pval_threshold)]
up = res_annot[(res_annot['log2FC'] >= log2FC_threshold) & (res_annot['FDR'] <= pval_threshold)]

# Same scatter plot as before with up- and down- regulated genes overlaid
#   in different colors

print('Number of up genes: '+str(up.shape[0]))
print('Number of down genes: '+str(down.shape[0]))

with PdfPages(gse_id+'_VolcanoUpDown_'+group2_label+'_vs_'+group1_label+'.pdf') as pdf:
    plt.figure(figsize=(6, 6))
    plt.scatter(x = res_annot['log2FC'], y = -np.log10(res_annot['FDR']), s = 1, color = 'lightgrey')
    plt.scatter(x = up['log2FC'], y = -np.log10(up['FDR']), s = 3, label = 'Up-regulated', color = 'red')
    plt.scatter(x = down['log2FC'], y = -np.log10(down['FDR']), s = 3, label = 'Down-regulated', color = 'green')
    plt.xlabel('log2FC')
    plt.ylabel('-log10 FDR')
    plt.axvline(-1*log2FC_threshold , color = 'grey', linestyle = '--')
    plt.axvline(log2FC_threshold,color = 'grey', linestyle = '--')
    plt.axhline(-np.log10(pval_threshold), color = 'grey', linestyle = '--')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

