#############################################################
## BME 598:  in_class_activity_8_1.py                      ##
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

## Import libraries
import GEOparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns


## Load up the GSE
gse = GEOparse.get_GEO("GSE75003")


## Get a list of GSMs
print(gse.gsms.keys())
print(gse.gsms["GSM1940293"].columns)


## Print out phenotypes
print(gse.phenotype_data)
print(gse.phenotype_data['characteristics_ch1.0.cell line'])
print(gse.phenotype_data['characteristics_ch1.1.sirna'])
convert_GSMs = gse.phenotype_data['characteristics_ch1.1.sirna'].to_dict()

## Create an expression matrix
# Note: This won't work well with lots of samples!
expr = gse.pivot_samples('VALUE')
print(expr)


## We want a specific genes:
#  - ETV6
#  - NFKB1
print(gse.gpls['GPL570'].table)
print(gse.gpls['GPL570'].table.columns)
probes = gse.gpls['GPL570'].table.loc[gse.gpls['GPL570'].table['Gene Symbol'].isin(['ETV6', 'NFKB1'])][['ID','Gene Symbol']]
probes = probes.set_index('ID')
probes = probes.to_dict()['Gene Symbol']
print(probes)


## Change expression data to long form
expr2 = expr
expr2.columns = [convert_GSMs[i] for i in expr2.columns]
expr2 = expr2.reset_index()
expr_melt = expr2.melt(id_vars='ID_REF')
print(expr_melt.head())


## Plot violin plots for each probe
with PdfPages('effect_of_siRNA_on_expression.pdf') as pdf:
    for probe1 in probes:
        if probe1 in expr.index:
            print(probe1+' '+probes[probe1])
            # Make violin and swarm plot
            fig, ax = plt.subplots(1,1)
            sns.boxplot(x='variable', y='value', hue='variable', data=expr_melt.loc[expr_melt['ID_REF']==probe1], ax = ax)
            plt.axhline(y=expr_melt.loc[(expr_melt['ID_REF']==probe1) & (expr_melt['variable']=='Control siRNA')]['value'].median(), color='red', linestyle='--', linewidth=2)
            plt.legend()
            sns.move_legend(obj=ax, loc="lower center", bbox_to_anchor=(.5, 1.1), ncol=2, title=None, frameon=False)
            ax.set_title(probes[probe1])
            plt.xticks(rotation=30)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()


