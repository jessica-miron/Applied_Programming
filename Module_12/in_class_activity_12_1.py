##########################################################
## BME 598:  in_class_activity_12_1.py                  ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed by: Plaisier Lab                          ##
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

## 1. Load up the packages you will need to run
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.datasets import make_classification
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


## 2. Generate synthetic data
X, y = make_classification(
    n_samples=1000,          # Number of samples
    n_features=2,            # Number of features
    n_informative=2,         # Number of informative features
    n_redundant=0,           # Number of redundant features
    n_repeated=0,            # Number of repeated features
    n_classes=2,             # Number of classes
    n_clusters_per_class=1,  # Number of clusters per class
    weights=[0.7, 0.3],      # Class distribution (imbalanced example)
    flip_y=0.01,             # Fraction of samples whose class is randomly flipped
    class_sep=1,           # Factor controlling the separation between classes
    random_state=42          # For reproducibility
)


# Convert to Pandas DataFrame for easier viewing and manipulation
df1 = pd.DataFrame(X)
df1['target'] = y

# Display the first few rows of the generated data
print("Generated Data Head:")
print(df1.head())

# Plot the data to visualize the classes
with PdfPages('two_classes.pdf') as pdf:
    plt.figure(figsize=(8, 6))
    plt.scatter(df1[0], df1[1], c=df1['target'], cmap=matplotlib.colors.ListedColormap(sns.color_palette('tab10')[0:2]), s=50, alpha=0.7)
    plt.title('Simulated Classification Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    pdf.savefig()
    plt.close()


## 1. Simulate data for a binary classification problem
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, n_classes=2, random_state=42)

print(f"Simulated data created with {X.shape[0]} samples and {X.shape[1]} features.")

## 2. Split the data into training and testing sets
# test_size: proportion of the dataset to include in the test split
# random_state: for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### KNN ###

# 3. Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3) # Choose an appropriate k value
knn.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred_knn = knn.predict(X_test)

## 5. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)

print(f"KNN Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


## Random Forest ##

## 3. Initialize and train a Random Forest Classifier
# n_estimators: number of trees in the forest
# random_state: for reproducibility
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

## 4. Make predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)

## 5. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Classifier Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


## Cross-validation ##

# CV KNN
cv_knn = cross_val_score(knn, X_train, y_train, cv=100)
print(f"KNN Classifier mean +/- sd: {np.mean(cv_knn):.4f} +\- {np.std(cv_knn):.4f}")

# CV RF
cv_rf = cross_val_score(rf_classifier, X_train, y_train, cv=100)
print(f"Random Forest Classifier mean +/- sd: {np.mean(cv_rf):.4f} +\- {np.std(cv_rf):.4f}")
