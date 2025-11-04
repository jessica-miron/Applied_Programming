#############################################################
## BME 598:  in_class_activity_11_2.py                     ##
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
import seaborn as sns

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats

from sklearn.datasets import make_blobs


## Return an eigengene for a gene expression data given a set of genes
def getEigengene(gexp, genes):
    pca = PCA(n_components=1)
    gexp_pca = pd.DataFrame(pca.fit(gexp.loc[genes].T).transform(gexp.loc[genes].T), index = gexp.columns)
    eigengene = gexp_pca[0]
    if sum([stats.pearsonr(gexp.loc[i],eigengene)[0] for i in genes])/len(genes) > 0:
        return eigengene
    else:
        return -eigengene


## Simulate some data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


## Iterate across possible k values, and compute inertia
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    inertias.append(km.inertia_)

with PdfPages('KMeans_elbow_plot.pdf') as pdf:
    plt.plot(range(1, 10), inertias, 'o-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    pdf.savefig()
    plt.close()


## Stadardize data
tmp = StandardScaler().fit_transform(X)

## Cluster using kMeans
sil_km = []
with PdfPages('km_silhouettes_simulated.pdf') as pdf:
    for i in range(2,10):
        n_clusters = i
        km1 = KMeans(n_clusters=i).fit(tmp)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        sil_km.append(silhouette_score(tmp, km1.labels_))

        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        #fig.set_size_inches(7, 7)

        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(tmp) + (n_clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tmp, km1.labels_)

        y_lower = 10
        for j in range(i):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            jth_cluster_silhouette_values = \
                sample_silhouette_values[km1.labels_ == j]

            jth_cluster_silhouette_values.sort()

            size_cluster_j = jth_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, jth_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=sil_km[i-2], color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Save figure to pdf
        pdf.savefig(fig)
        plt.close()

    # Save plots to pdf
    fig = plt.figure()
    plt.plot(range(2,10),sil_km)
    plt.xticks(range(2,10))
    plt.xlabel('Number of clusters')
    plt.ylabel('Average sihouette score')
    pdf.savefig(fig)
    plt.close()



########################
## Real world dataset ##
########################

## Load up real-world data into pandas
# Using data from GSE79731 on GEO:  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE79731
# A study of Mycobacterium tuberculosis infection


## Load phenotype data
phenos = pd.read_csv('GSE79731_phenos.csv', header = 0, index_col = 0)


## Load up the GSE
gse = GEOparse.get_GEO("GSE79731")


## Get a list of GSMs
print(gse.gsms.keys())
print(gse.gsms["GSM2101737"].columns)


## Print out phenotypes
print(gse.phenotype_data)
print(gse.phenotype_data['characteristics_ch1.3.infection'])
print(gse.phenotype_data['characteristics_ch1.4.timepoint'])
convert_GSMs = gse.phenotype_data['characteristics_ch1.4.timepoint'].to_dict()

## Create an expression matrix
# Note: This won't work well with lots of samples!
expr = gse.pivot_samples('VALUE')
print(expr)


## We want a specific genes:
#  Need to use mygene to convert entrez IDs to gene symbols
print(gse.gpls['GPL13712'].table)
print(gse.gpls['GPL13712'].table.columns)
## Uh oh! No conversion to gene symbol.


## Load gene info - separated by tabs not commas, and index_col is set to 1 not 0
# https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/
gene_info = pd.read_csv('Mus_musculus.gene_info', sep='\t', header = 0, index_col = 1)


## Translate Entrez gene ids to gene symbols
expr2 = expr

# Remove _at and make Entrez ID into an integer
expr2.index = [int(i.split('_')[0]) for i in expr2.index]

# Remove genes which don't have a gene symbol
expr2 = expr.loc[expr.index.isin(gene_info.index)]

# Map symbols onto index
expr2.index = gene_info.loc[expr2.index,'Symbol']


## Use timepoint as column IDs
expr2.columns = [convert_GSMs[i] for i in expr2.columns]


## Feature selection
top3000 = expr2.var(axis=1).sort_values(ascending=False).index[range(3000)]


## Scaling
tmp = StandardScaler().fit_transform(expr2.loc[top3000])


## Cluster using kMeans
sil_km = []
with PdfPages('km_silhouettes.pdf') as pdf:
    for i in range(2,20):
        n_clusters = i
        km1 = KMeans(n_clusters=i).fit(tmp)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        sil_km.append(silhouette_score(tmp, km1.labels_))

        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        #fig.set_size_inches(7, 7)

        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, (len(tmp) + (n_clusters + 1) * 10)])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tmp, km1.labels_)

        y_lower = 10
        for j in range(i):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            jth_cluster_silhouette_values = \
                sample_silhouette_values[km1.labels_ == j]

            jth_cluster_silhouette_values.sort()

            size_cluster_j = jth_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, jth_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=sil_km[i-2], color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Save figure to pdf
        pdf.savefig(fig)
        plt.close()

    # Save plots to pdf
    fig = plt.figure()
    plt.plot(range(2,20),sil_km)
    plt.xticks(range(2,20))
    plt.xlabel('Number of clusters')
    plt.ylabel('Average sihouette score')
    pdf.savefig(fig)
    plt.close()

# Chose k = 4
km1 = KMeans(n_clusters = 4).fit(tmp)
print(km1.labels_)
eigengenes = pd.concat([getEigengene(expr2.loc[top3000], top3000[km1.labels_==i]) for i in range(len(set(km1.labels_)))], axis = 1)
eigengenes.columns = range(len(set(km1.labels_)))

# Make column and row colors
colors = {'0h':'#fee5d9', '4h':'#fcae91', '8h':'#fb6a4a', '24h':'#de2d26', '48h':'#a50f15'}
colRow_colors = [colors[i] for i in phenos.loc['timepoints']]

# Plot clustermap
sns.clustermap(eigengenes.T, cmap = sns.color_palette("vlag",n_colors=33), col_colors=colRow_colors, col_cluster=False)
plt.show()

# Make it into a PDF
with PdfPages('GSE79731_clustered_KMeans_4.pdf') as pdf:
    # Plot clustermap
    sns.clustermap(eigengenes.T, cmap=sns.color_palette('vlag', n_colors=33), col_colors=colRow_colors, col_cluster=False)
    pdf.savefig()
    plt.close()

