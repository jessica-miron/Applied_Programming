#############################################################
## BME 598:  in_class_activity_7_2.py                      ##
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

## Total points = 100 pts (70 pts through autograder, 30 pts through PDF submitted through Canvas)


## Your name
# Jessica Miron 1224633969



## 1. (10pts) Prepare an executive summary that clearly states the goals of the proposed analysis and provides a concise overview of the analytical steps to be undertaken 
#             A. The executive summary needs to be written in plain text; not as bullet points. And should be no more than one paragraph.
#             B. The overview of analytical steps should be written as a outline with bullet points. Provide more detail than the headings of the Deliverables section.




## 2. (10pts) Load up the packages you will need to run
#  Deliverables:
#     a. Restart console: Consoles tab -> Restart Kernel
#     b. Import all the packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


## 3. (10pts) Load up and inspect data and function
#  Deliverables:
#     a. Use a text editor to determine the structure of the 'Raw_data_TRP_with_ischemic_stroke.csv' file
#        - Delimiter?
#        - Header?
#        - Index column?
#     c. Read the file using pandas into a variable named 'stroke'
#     d. Make sure the shape of the file is correct
#        - Expectation is (462 rows, by 10 columns)

stroke = pd.read_csv('Raw_data_TRP_with_ischemic_stroke.csv',header=0)
print(stroke.shape)


## 4. (10pts) Make a pairs plot of your data to identify which variables to test in correlation studies.
#  Deliverables:
#     a. Write a pairsplot PDF using PdfPages to a pdf called 'stroke_data_pairplot.pdf'

with PdfPages('stroke_data_pairplot.pdf') as pdf:
    plt.figure(layout='constrained')
    sns.pairplot(data=stroke)
    pdf.savefig()
    plt.close()

## 5. (20pts) Compute pairwise correlations
#  Deliverables:
#     a. Make a variable 'quant_variables' that is a list of all the columns that should be inlcuded in
#        the corrleation studies
#     b. Make two dictionaries to hold the results of correlation analysis:
#        - 'pearson_cor' to hold the correlation values
#        - 'pearsons_pv' to hold the p-values
#     c. Fill the dictionaries up with the values from the pairwise correlation comparisons of the 'quant_variables'
#        - NOTE: set the comparison of a variable to itself as 0

dropped_na = stroke.dropna(axis=0)

quant_variables = ['Age', 'Sex', 'Smoking', 'SBP', 'DBP', 'HbA1c', 'FBS', 'TC', 'LDL',
       'HDL', 'TG', 'WBC', 'CRP', 'AIP', 'TyG']
pearson_cor = {}
pearson_pv = {}
for q1 in quant_variables:
    pearson_cor[q1] = {}
    pearson_pv[q1] = {}
    for q2 in quant_variables:
        if not q1==q2:
            pearson_cor[q1][q2], pearson_pv[q1][q2] = stats.pearsonr(dropped_na[q1], dropped_na[q2])
        else:
            pearson_cor[q1][q2], pearson_pv[q1][q2] = (0, 0)
# Convert to pandas DataFrames
pearson_cor_df = pd.DataFrame(pearson_cor)
pearson_cor_df_plot = pearson_cor_df.copy()
pearson_cor_df.columns = pearson_cor_df.columns+'.R'
pearson_pv_df = pd.DataFrame(pearson_pv)
pearson_pv_df.columns = pearson_pv_df.columns+'.pv'
# Write out combined R and PV matrix
combined_df = pd.concat([pearson_cor_df, pearson_pv_df], axis=1).sort_index(axis=1)
combined_df.to_csv('stroke_pearson_correlation.csv')

## 6. (10pts) Convert result dictionaries into pd.DataFrames and write them out as CSV files
#  Deliverables:
#     a. Convert both result dictionaries into pd.DataFrames:
#        - 'pearson_cor' -> 'pearson_cor_df'
#        - 'pearson_pv' -> 'pearson_pv_df'
#     b. Concatenate the two pd.DataFrames and change the names of the columns to denote correlation coefficient 'R',
#        and p-value '.pv'
#     c. Then write out the merged pd.DataFrame as 'stroke_pearson_correlation.csv'




## 7. (10pts) Visiualize the correlation matrix as a heatmap
#  Deliverables:
#     a. Generate a heatmap as a PDF called 'stroke_pearson_cor_df_heatmap.pdf'
#        - NOTE: Save the sns.heatmap to a variable called 'hm1' to facilitate the autograding, i.e.:
#                hm1 = sns.heatmap(<fill with your parameters!>)
#     b. Use the 'vlag' cmap which goes from red to white to blue
#     c. Maximum and minimum correlation values should be set at 1 and -1, respectively
#     d. Must include units for the colorbar legend

with PdfPages('stroke_pearson_cor_df_heatmap.pdf') as pdf:
    plt.figure(layout='constrained')
    hm1 = sns.heatmap(pearson_cor_df_plot, cmap=sns.color_palette("vlag", as_cmap=True),vmin=-1,vmax=1,cbar_kws={'label': 'Pearson\'s R'})
    plt.title('Pearson Correlation Heatmap for Stroke Dataset')
    plt.xticks(rotation=50)
    pdf.savefig()
    plt.close()

## 8. (20pts) Biological interpretation that is to be completed in a separate Word document that will be converted into a PDF for submission
#  Deliverables:
#     a. Using a minimum of 3 and a maximum of 5 paragraphs of plain text (not bullet points), describe how the data in the figures, and
#        output from statistical tests in the results tables answer the following questions:
#        - How many relationships were observed to be significant?
#            - How many were positive?
#            - How many were negative?
#        - What is the meaning of the relationships?
#        - Any trends that could reflect tumor biology?
#        - Are the relationships known? Do some research to validate at least two relationships using one citation.
#     c. Please embed the figures and excerpts of the table to reinforce your interpretation.




