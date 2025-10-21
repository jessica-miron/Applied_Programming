##########################################################
## BME 494/598:  Homework 6 Python code                 ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed for: BME 494/598 Applied Programming      ##
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

## Total points = 100 pts



## Your name
# Jessica Miron 1224633969


## 1. (20pts) Load up the packages you will need to run
#  Deliverables:
#     a. Restart console: Consoles tab -> Restart Kernel
#     b. Import all the packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages

## 2. (10pts) Load up and inspect data and function
#  Deliverables:
#     a. Use a text editor to determine the structure of the 'proliferation_v3_w_redos_clean.csv' file
#        - Delimiter?
#        - Header?
#        - Index column?
#     c. Read the file using pandas into a variable named 'prolif'
#     d. Make sure the shape of the file is correct
#        - Expectation is (42 rows, by 14 columns)
prolif = pd.read_csv('proliferation_v3_w_redos_clean.csv',header=0,index_col=0)
print(prolif.shape)

## 3. (15pts) Background subtraction:
#  Deliverables:
#         The results for these proliferation assays were conducted on different plates, and therefore
#     the No_cell values will vary across plates. Thus, we must compute these for the cell line and
#     plate on which it was run. The different names in the No_cell column of prolif correspond to the
#     different plates, and the specific cell line it was compute for are in the columns labeled by
#     the cell line replicates (*.1, *.2, *.3, *.4).
#     a. Compute the median 'No cell' background for each cell line and store in a nested dictionary
#        named 'median_no_cell' that has name of the 'No_cell' row name and then the name of each cell
#        line under that:
#            Example:
#
#                median_no_cell = {'No_cell': {'HA':1234, 'T98G': 1234, 'U251': 1234},
#                                   ...
#                                 }
#
#     b. Then, subtract the median 'No cell' background value from each perturbation from the same cell
#        line. Make a new pandas DataFrame with corrected cell counts called 'prolif_corrected'
HA = ['HA'+'.'+str(i) for i in range(1,5)]
T98G = ['T98G'+'.'+str(i) for i in range(1,5)]
U251 = ['U251'+'.'+str(i) for i in range(1,5)]
no_cell_names = ['No_Cell','No_Cell.2','No_Cells_Vehicle.3']
median_no_cells={}
for i in range(0,3):
    HA_values=[]
    T98G_values=[]
    U251_values=[]
    for j in range(0,4):
        HA_values.append(int(prolif.loc[no_cell_names[i],HA[j]]))
        T98G_values.append(int(prolif.loc[no_cell_names[i],T98G[j]]))
        U251_values.append(int(prolif.loc[no_cell_names[i],U251[j]]))
    buffer = {'HA':np.median(HA_values),'T98G':np.median(T98G_values),'U251':np.median(U251_values)}
    median_no_cells[no_cell_names[i]]=buffer
print(median_no_cells)
prolif_corrected = prolif.copy()
for index, row in prolif_corrected.iterrows():
    no_cell_value=row['No_cell']
    if no_cell_value in no_cell_names:
        median_values = median_no_cells[no_cell_value]
        for col_name, value in row.items():
            if col_name.startswith(('HA.','T98G.','U251.')):
                cell_name=col_name[0:-2]
                subtraction_value = median_values[cell_name]
                prolif_corrected.loc[index,col_name]=value - subtraction_value
print(prolif_corrected.head(10))

## 4. (15pts) Plot proliferation per treatment and cell line
#  Deliverables:
#     a. Convert 'prolif_corrected' into long format with column for cell line, perturbation, and proliferation, i.e.:
#        Example:
#
#            cell line,perturbation,proliferation
#            T98G.1,miR-34a mimic,1234
#            ...
#
#     b. Make three plots, one for each cell line, with violin plots for each perturbation
#     c. Be sure to include legend, axes labels, and title
#     d. Save as all plots into one PDF called 'perturbation_by_cell_line.pdf'
new_prolif = prolif_corrected.copy()
new_prolif = new_prolif.drop(['No_Cell','No_Cell.2','No_Cells_Vehicle.3'])
new_prolif_part4 = new_prolif.reset_index(names='perturbation')
prolif_columns = new_prolif_part4.columns
new_column_names = []
for i in range(len(prolif_columns)):
    if prolif_columns[i].startswith(('HA.','T98G.','U251.')):
        new_name = prolif_columns[i][0:-2]
        new_column_names.append(new_name)
    else:
        new_column_names.append(prolif_columns[i])
new_prolif_part4.columns = new_column_names
prolif_plot_long = pd.melt(new_prolif_part4, id_vars='perturbation',value_vars=['HA','T98G','U251'])
print(prolif_plot_long.head(15))

prolif_plot_long_HA = prolif_plot_long.loc[prolif_plot_long['variable']=='HA']
prolif_plot_long_T98G = prolif_plot_long.loc[prolif_plot_long['variable']=='T98G']
prolif_plot_long_U251 = prolif_plot_long.loc[prolif_plot_long['variable']=='U251']

with PdfPages('violinPlots_byCellLine_byPerturbation.pdf') as pdf:
    plt.figure(figsize=(8,5),layout='constrained')
    sns.violinplot(x='perturbation',y='value',data=prolif_plot_long_HA)
    plt.xlabel('Perturbation')
    plt.ylabel('Cell Proliferation Value')
    plt.title('Cell Proliferation by Perturbation for HA Cell Line')
    plt.xticks(rotation=90)
    pdf.savefig()
    plt.close()
    plt.figure(figsize=(8,5),layout='constrained')
    sns.violinplot(x='perturbation',y='value',data=prolif_plot_long_T98G)
    plt.xlabel('Perturbation')
    plt.ylabel('Cell Proliferation Value')
    plt.title('Cell Proliferation by Perturbation for T98G Cell Line')
    plt.xticks(rotation=90)
    pdf.savefig()
    plt.close()
    plt.figure(figsize=(8,5),layout='constrained')
    sns.violinplot(x='perturbation',y='value',data=prolif_plot_long_U251)
    plt.xlabel('Perturbation')
    plt.ylabel('Cell Proliferation Value')
    plt.title('Cell Proliferation by Perturbation for U251 Cell Line')
    plt.xticks(rotation=90)
    pdf.savefig()
    plt.close()

## 5. (20pts) Compare each treatment versus its corresponding control per cell line and per perturbation
#  Deliverables:
#     a. Set up a dictionary to hold the output of the comparisons called 'perturbation_comparisons'
#         - Hint: Need to capture cell line, perturbation, fold-change, T-statistic, and p-value
#        Example:
#                              HA.FC    HA.stat  HA.pvalue   T98G.FC  T98G.stat   T98G.pvalue   U251.FC  U251.stat   U251.pvalue
#         CEBPD_siRNA       0.670732  -2.104883   0.079936  0.836220  -3.646332  1.075329e-02  0.891289  -4.541457  3.927184e-03
#         miR_29a_Mimic     1.433333   2.861776   0.028735  1.208154   2.326915  5.888886e-02  1.385634   4.243691  5.417672e-03
#                                                                   ...
#
#        * Let pandas decide if scientific notation is needed. In this case p-values for T98G and U251 are in scientifc notation
#        - For the column names use [HA, T98G, U251] connected to [FC, stat, pvalue] using a period, e.g. [HA.FC, HA.stat, HA.pvalue]
#
#     b. Extract the proliferation values for the four replicates for the perturbation of interest
#     c. Extract the proliferation values for the four replicates for the corresponding negative control
#     d. Compute the fold-change as the median of the perturbation proliferation values divided by the
#        median of proliferation for the negative control, and store it into the dictionary
#     e. Compute the Student's t-test p-value using the replicate values for the perturbation versus the
#        replicate values for the negative control (don't use the median value, use four replicate
#        values), and store into the dictionary
#     f. Convert the dictionary into a pandas DataFrame called 'perturbation_comparisons_df'
#     g. Write out the DataFrame to a csv file called 'perturbation_comparisons_df.csv'
perturbation_comparisons = {}
header = ['HA.FC','HA.stat','HA.pvalue','T98G.FC','T98G.stat','T98G.pvalue','U251.FC','U251.stat','U251.pvalue']
perturbation_comparisons['']=header
print(perturbation_comparisons)
"""
read 3 no cell lines and get groups (in loop for three groups)
if no_cell_value equals no_cell we are on then get out cell values
then compute and add to dictionary. Move through perturbations. Move through no_cell
"""
neg_control_names = ['NC_Mimic', 'NC_Mimic.2', 'NC_siRNA', 'V.DMSO']
prolif_dropped = new_prolif.drop(neg_control_names)
for i in range(0,4):
    HA_values_neg_control=[]
    T98G_values_neg_control=[]
    U251_values_neg_control=[]
    for j in range(0,4):
        HA_values_neg_control.append(int(new_prolif.loc[neg_control_names[i],HA[j]]))
        T98G_values_neg_control.append(int(new_prolif.loc[neg_control_names[i],T98G[j]]))
        U251_values_neg_control.append(int(new_prolif.loc[neg_control_names[i],U251[j]]))
    for index, row in prolif_dropped.iterrows():
        neg_control_value=row['Negative_control']
        if neg_control_value == neg_control_names[i]:
            HA_values=[]
            T98G_values=[]
            U251_values=[]
            for k in range(0,4):
                HA_values.append(row[HA[k]])
                T98G_values.append(row[T98G[k]])
                U251_values.append(row[U251[k]])
            HA_FC=np.median(HA_values)/np.median(HA_values_neg_control)
            HA_stat, HA_pvalue = stats.ttest_ind(HA_values, HA_values_neg_control)
            T98G_FC=np.median(T98G_values)/np.median(T98G_values_neg_control)
            T98G_stat, T98G_pvalue = stats.ttest_ind(T98G_values, T98G_values_neg_control)
            U251_FC=np.median(U251_values)/np.median(U251_values_neg_control)
            U251_stat, U251_pvalue = stats.ttest_ind(U251_values, U251_values_neg_control)
            buffer = [HA_FC,HA_stat,HA_pvalue,T98G_FC,T98G_stat,T98G_pvalue,U251_FC,U251_stat,U251_pvalue]
            perturbation_comparisons[index]=buffer
print(perturbation_comparisons)
perturbation_comparisons_df = pd.DataFrame(perturbation_comparisons)
perturbation_comparisons_df = perturbation_comparisons_df.transpose()
new_header = perturbation_comparisons_df.iloc[0]
perturbation_comparisons_df = perturbation_comparisons_df[1:]
perturbation_comparisons_df.columns = new_header
perturbation_comparisons_df.to_csv('perturbation_comparisons_df.csv')
## 6. (20pts) Biological interpretation that is to be completed in a separate Word document that will be converted into a PDF for submission
#  Deliverables:
#     HINT: These responses are to be completed in a separate Word document that will be converted into a PDF for submission
#     a. Which many treatments increased or decreased proliferation?
#     b. Which differences are statistically significant?
#     c. What are the differences between the human astrocyte (HA) and glioblastoma lines (T98G and U251)?
#     d. Any trends that could reflect tumor biology?
#     e. Which combinations from Table S19 listed above should be prioritized?
#     c. Be sure to include legend, axes labels, and title
#     d. Save as all plots into one PDF called 'perturbation_by_cell_line.pdf'

