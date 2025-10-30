#############################################################
## BME 598:  in_class_activity_11_1.py                     ##
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


#################################
## Step 1. What is clustering? ##
#################################

import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.datasets import make_blobs

# Simulate some data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

# Plot the data
with PdfPages('how_many_clusters.pdf') as pdf:
    plt.scatter(X[:, 0], X[:, 1], s=30)
    plt.title("How many clusters do you see?")
    plt.xlabel('Arbitrary units')
    plt.ylabel('Arbitrary units')
    pdf.savefig()
    plt.close()


# Plot the data with colors for clusters
colors1 = {0:'blue', 1:'red', 2:'green', 3:'orange'}

with PdfPages('how_many_clusters_trueLabels.pdf') as pdf:
    plt.scatter(X[:, 0], X[:, 1], s=30, color=[colors1[i] for i in y_true])
    plt.title("How many clusters do you see?")
    plt.xlabel('Arbitrary units')
    plt.ylabel('Arbitrary units')
    pdf.savefig()
    plt.close()


################################################
## Step 2. What is clustering:  Agglomerative ##
################################################

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Compute linkage matrix using ward algorithm
Z = linkage(X, method="ward")

# Plot linkage matrix as dendrogram
with PdfPages('hierarchical_clustering_dendrogram.pdf') as pdf:
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    pdf.savefig()
    plt.close()

# Plot linkage matrix as dendrogram with cutoff
with PdfPages('hierarchical_clustering_dendrogram_wClusters.pdf') as pdf:
    labels = fcluster(Z, t=4, criterion='maxclust')
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=30)
    plt.title("Clusters from Hierarchical Clustering")
    plt.xlabel('Arbitrary units')
    plt.ylabel('Arbitrary units')
    pdf.savefig()
    plt.close()


#########################################
## Step 3. What is clustering:  Kmeans ##
#########################################

from sklearn.cluster import KMeans

# Conduct KMeans clustering on data
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# Plot the labels
with PdfPages('KMeans_clustering.pdf') as pdf:
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=30)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='x')
    plt.title("K-Means Clustering")
    plt.xlabel('Arbitrary units')
    plt.ylabel('Arbitrary units')
    pdf.savefig()
    plt.close()


#################################
## Step 4. Methods to define k ##
#################################

import numpy as np

# Iterate across possible k values, and compute inertia
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


###############################
## Step 5. Comparing methods ##
###############################

from sklearn.metrics import adjusted_mutual_info_score

print(adjusted_mutual_info_score(y_true, fcluster(Z, t=4, criterion='maxclust')))
print(adjusted_mutual_info_score(y_true, kmeans.labels_))

