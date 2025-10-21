##########################################################
## BME 494/598:  In-class activity 7-1                  ##
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

# Import libraries
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

# Load up data
sleep_data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv', header=0, index_col=0)
sleep_data.info()

## Plot histograms of quantitative and qualitative data
with PdfPages('sleep_data_hist.pdf') as pdf:
    # Pairs plot of quantitative data
    plt.figure(figsize=(3, 3))
    sns.histplot(data=sleep_data, x='Sleep Duration')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Pairs plot of qualitative data
    plt.figure(figsize=(3, 3))
    ax = sns.histplot(data=sleep_data, x='Sex')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Pairs plot of qualitative data
    plt.figure(figsize=(3, 3))
    ax = sns.histplot(data=sleep_data, x='Sex', hue='Sex', discrete=True, shrink=0.8)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


## Let's pull out some relationships
with PdfPages('sleep_data_jointplot.pdf') as pdf:
    # Plotting Age by Sleep Durtion colored by Sex
    plt.figure(figsize=(6, 6))
    sns.jointplot(data=sleep_data, x="Age", y="Sleep Duration", hue="Sex")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plotting Heart Rate by Daily Steps colored by BMI Category
    plt.figure(figsize=(6, 6))
    sns.jointplot(data=sleep_data, x="Heart Rate", y="Daily Steps", hue="BMI Category")
    plt.tight_layout()
    pdf.savefig()
    plt.close()


## Plot the quantitative data against one another:
#   Age, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, Heart Rate, Daily Steps
with PdfPages('sleep_data_pairplot.pdf') as pdf:
    # Pairs plot of quantitative data
    plt.figure(figsize=(6, 6))
    sns.pairplot(data=sleep_data)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Pairs plot of quantitative data split by Sex
    plt.figure(figsize=(6, 6))
    sns.pairplot(data=sleep_data, hue='Sex')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Pairs plot of quantitative data split by Occupation
    plt.figure(figsize=(6, 6))
    sns.pairplot(data=sleep_data, hue='Occupation')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Pairs plot of quantitative data split by BMI Category
    plt.figure(figsize=(6, 6))
    sns.pairplot(data=sleep_data, hue='BMI Category')
    plt.tight_layout()
    pdf.savefig()
    plt.close()


## Compute pair-wise correlations
quant_variables = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
pearson_cor = {}
pearson_pv = {}
for q1 in quant_variables:
    pearson_cor[q1] = {}
    pearson_pv[q1] = {}
    for q2 in quant_variables:
        if not q1==q2:
            pearson_cor[q1][q2], pearson_pv[q1][q2] = pearsonr(sleep_data[q1], sleep_data[q2])
        else:
            pearson_cor[q1][q2], pearson_pv[q1][q2] = (np.nan, np.nan)

# Convert to pandas DataFrames
pearson_cor_df = pd.DataFrame(pearson_cor)
pearson_pv_df = pd.DataFrame(pearson_pv)

# Write out combined R and PV matrix
combined_df = pd.concat([pearson_cor_df, pearson_pv_df], axis=1).sort_index(axis=1)
combined_df.columns = [combined_df.columns[i]+'.pv' if ((i+1)%2)==0 else combined_df.columns[i]+'.R' for i in range(len(combined_df.columns))]
combined_df.to_csv('pearson_correlation.csv')

# Make a plot of the relationships
with PdfPages('pearson_cor_df_heatmap.pdf') as pdf:
    plt.figure(figsize=(6, 6))
    sns.heatmap(data=pearson_cor_df, cmap=sns.color_palette("vlag", as_cmap=True), vmin=-1, vmax=1, cbar_kws={'label': 'Pearson\'s R'})
    plt.tight_layout()
    pdf.savefig()
    plt.close()


## Compute non-parametric pair-wise correlations
quant_variables = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
spearman_cor = {}
spearman_pv = {}
for q1 in quant_variables:
    spearman_cor[q1] = {}
    spearman_pv[q1] = {}
    for q2 in quant_variables:
        if not q1==q2:
            spearman_cor[q1][q2], spearman_pv[q1][q2] = spearmanr(sleep_data[q1], sleep_data[q2])
        else:
            spearman_cor[q1][q2], spearman_pv[q1][q2] = (np.nan, np.nan)

# Convert to pandas DataFrames
spearman_cor_df = pd.DataFrame(spearman_cor)
spearman_pv_df = pd.DataFrame(spearman_pv)

# Write out combined R and PV matrix
combined_df = pd.concat([spearman_cor_df, spearman_pv_df], axis=1).sort_index(axis=1)
combined_df.columns = [combined_df.columns[i]+'.pv' if ((i+1)%2)==0 else combined_df.columns[i]+'.R' for i in range(len(combined_df.columns))]
combined_df.to_csv('spearman_correlation.csv')

# Make a plot of the relationships
with PdfPages('spearman_cor_df_heatmap.pdf') as pdf:
    plt.figure(figsize=(6, 6))
    sns.heatmap(data=spearman_cor_df, cmap=sns.color_palette("vlag", as_cmap=True), vmin=-1, vmax=1, cbar_kws={'label': 'Spearman\'s rho'})
    plt.tight_layout()
    pdf.savefig()
    plt.close()


############################################
### Example with non-linear relationship ###
############################################

# Simple example
x1 = list(range(1,100))
y1 = x1
y2 = np.exp(x1)

# Plot both on a scatterplot
with PdfPages('simple_non_linear_example_scatter.pdf') as pdf:
    fig, ax = plt.subplots(1, 2, figsize = (6,3))
    sns.scatterplot(x = x1, y = y1, ax = ax[0])
    ax[0].set(xlabel="x1", ylabel="y1")
    sns.scatterplot(x = x1, y = y2, ax = ax[1])
    ax[1].set(xlabel="x1", ylabel="y2")
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    fig, ax = plt.subplots(1, 2, figsize = (6,3))
    sns.regplot(x = x1, y = y1, ax = ax[0], line_kws={'color': 'red', 'linewidth': 1.5})
    ax[0].set(xlabel="x1", ylabel="y1")
    sns.regplot(x = x1, y = y2, ax = ax[1], line_kws={'color': 'red', 'linewidth': 1.5})
    ax[1].set(xlabel="x1", ylabel="y2")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# Correlation of first plot with perfect linear fit
print(pearsonr(x1, y1))
print(spearmanr(x1, y1))

# Correlation of second plot with poor linear fit
print(pearsonr(x1, y2))
print(spearmanr(x1, y2))
