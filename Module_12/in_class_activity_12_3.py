##########################################################
## OncoMerge:  in_class_activity_12_3.py                ##
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


## May need to install
# pip install GEOparse
# pip install tensorflow


## 1. Load libraries
import GEOparse as gp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score
)


import random
random.seed(42)

np.random.seed(42)
tf.random.set_seed(42)

#%%
####################################################################
## Load up real-world data: Human whole blood microarray study    ##
## to compare patients with tuberculosis, sarcoidosis,            ##
## pneumonia, and lung cancer                                     ##
## PMID = 23940611                                                ##
## Note:  The authors have split their data into train, test,     ##
## and validation. But as we will learn using cross-validation    ##
## is a better approach. We will try both methods.                ##
####################################################################

## 2. Load up real-world data into pandas
# Using data from super-series GSE42834 on GEO:
#  - Training = https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42830
#  - Test = https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42826
#  - Validation = https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42825
gseTrain = gp.get_GEO(filepath="GSE42830_family.soft.gz")
gseTest = gp.get_GEO(filepath="GSE42826_family.soft.gz")
gseValidation = gp.get_GEO(filepath="GSE42825_family.soft.gz")


## 3. Extract phenotypes/labels (disease state in this case are the labels)
## For starters let's include four disease states:  Control, Active Sarcoid, TB, and Non-active sarcoidosis
print(gseTrain.phenotype_data)

# Select out gender, ethnicity, and disease state
phenosTrain = gseTrain.phenotype_data[['characteristics_ch1.0.gender','characteristics_ch1.1.ethnicity','characteristics_ch1.2.disease state']]
phenosTrain.columns = ['gender','ethnicity','disease_state']
phenosTrain

# Get rid of cacner and 
subset_train = phenosTrain.index[phenosTrain['disease_state'].isin(['Control','Active Sarcoid','TB','Non-active sarcoidosis'])]
phenosTrain = phenosTrain.loc[subset_train]

# Number of each disease_state in GSE42830
print(phenosTrain['disease_state'].value_counts())

# Phenotypes for test dataset (GSE42826)
phenosTest = gseTest.phenotype_data[['characteristics_ch1.0.gender','characteristics_ch1.1.ethnicity','characteristics_ch1.2.disease state']]
phenosTest.columns = ['gender','ethnicity','disease_state']
print(phenosTest['disease_state'].value_counts())

# Get rid of cacner and 
subset_test = phenosTest.index[phenosTest['disease_state'].isin(['Control','Active Sarcoid','TB','Non-active sarcoidosis'])]
phenosTest = phenosTest.loc[subset_test]

# Phenotypes for validation dataset (GSE42825)
phenosValidation = gseValidation.phenotype_data[['characteristics_ch1.0.gender','characteristics_ch1.1.ethnicity','characteristics_ch1.2.disease state']]
phenosValidation.columns = ['gender','ethnicity','disease_state']
print(phenosValidation['disease_state'].value_counts())
# Uh-oh! Active sarcoidosis does not equal Active Sarcoid, from GSE42830 and GSE42826
# Let's fix it by harmonizing the validation to Active Sarcoid
phenosValidation.loc[phenosValidation.disease_state=='Active sarcoidosis','disease_state'] = 'Active Sarcoid'
print(phenosValidation['disease_state'].value_counts())

subset_validation = phenosValidation.index[phenosValidation['disease_state'].isin(['Control','Active Sarcoid','TB','Non-active sarcoidosis'])]
phenosValidation = phenosValidation.loc[subset_validation]

#%%
## 4. Extract gene expression data
# What columns are available per sample
# print(gseTrain.gsms['GSM1050928'].columns)
# We want to have VALUE for each sample combined into one dataframe

# Build a DataFrame from VALUEs using the pivot_samples function
# With genes as the rows and samples as the columns
gexpTrain = gseTrain.pivot_samples('VALUE').loc[:,subset_train]
gexpTest = gseTest.pivot_samples('VALUE').loc[:,subset_test]
gexpValidation = gseValidation.pivot_samples('VALUE').loc[:,subset_validation]
"""
qt = QuantileTransformer()
gexpTrainBuffer = qt.fit_transform(gexpTrain.T)
gexpTestBuffer = qt.transform(gexpTest.T)
gexpValidationBuffer = qt.transform(gexpValidation.T)
gexpTrain = pd.DataFrame(gexpTrainBuffer.T, columns=gexpTrain.columns, index=gexpTrain.index)
gexpTest = pd.DataFrame(gexpTestBuffer.T, columns=gexpTest.columns, index=gexpTest.index)
gexpValidation = pd.DataFrame(gexpValidationBuffer.T, columns=gexpValidation.columns, index=gexpValidation.index)
"""


#5 Feature Selection

topFeatures = gexpTrain.var(axis=1).sort_values(ascending=False).index[range(1000)]

#6 Scaling
scaler = StandardScaler()
gexpTrain_scaled = scaler.fit_transform(gexpTrain.loc[topFeatures].T)
gexpTest_scaled = scaler.fit_transform(gexpTest.loc[topFeatures].T)
gexpValidation_scaled = scaler.fit_transform(gexpValidation.loc[topFeatures].T)
"""
gexpFull = np.concatenate((gexpTrain, gexpValidation),axis=0)
gexpFull = pd.DataFrame(gexpFull)
scaler = StandardScaler()
gexpFull_scaled = scaler.fit_transform(gexpFull.loc[topFeatures].T)
gexpTest_scaled  = scaler.transform(gexpTest.loc[topFeatures].T)
"""

convertMe = {'TB':'TB','Active Sarcoid':0,'Non-active sarcoidosis':0,'Control':0}
y_train = pd.Series([1 if i=='TB' else 0 for i in [convertMe[i] for i in phenosTrain['disease_state']]])
y_test = pd.Series([1 if i=='TB' else 0 for i in [convertMe[i] for i in phenosTest['disease_state']]])
y_validation = pd.Series([1 if i=='TB' else 0 for i in [convertMe[i] for i in phenosValidation['disease_state']]])
#y_train = ([convertMe[i] for i in phenosTrain['disease_state']])
#y_test = ([convertMe[i] for i in phenosTest['disease_state']])
#y_validation = ([convertMe[i] for i in phenosValidation['disease_state']])

#gexpFull_scaled = np.concatenate((gexpTrain_scaled,gexpValidation_scaled),axis=0)
#y_full = np.concatenate((y_train,y_validation),axis=0)
#%%
#7 Build model
model = models.Sequential([layers.Input(shape=(gexpTrain_scaled.shape[1],)),
                           layers.Dense(600, activation='relu'),
                           layers.Dropout(0.3),
                           layers.Dense(200, activation='relu'),
                           layers.Dropout(0.2),
                           # Binary classification = sigmoid, multiple class classification = softmax
                           layers.Dense(1, activation='sigmoid')
                           ])

model.summary()


#class_weights = {0:(len(y_full)/(2*(len(y_full)-sum(y_full)))),1:(len(y_full)/(2*sum(y_full)))}
## 8. Compile
model.compile(
    optimizer=optimizers.SGD(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


## 9. Training the model
history = model.fit(
    gexpTrain_scaled, y_train,
    validation_split=0.15,
    epochs=3500,
    batch_size=32,
    verbose=2,
    #callbacks = [learning_rate],
    #class_weight = class_weights
)




#%%

"""
Changed to binary classifier, TB=0, everything else = 1
Added validation data to training data

1000 Features

Two layers: (neurons, dropout, neurons, dropout)
    600,0.3,200,0.2, min val loss = 0.1847
    600,0.3,200,0.3, min val loss = 0.1160
    600,0.3,200,0.4, min val loss = 0.1080
    600,0.3,200,0.5, min val loss = 0.1186
    600,0.4,200,0.4, min val loss = 0.2054
    600,0.2,200,0.4, min val loss = 0.0805 **
    600,0.1,200,0.4, min val loss = 0.1576
    600,0.2,150,0.4, min val loss = 0.2048
    600,0.2,250,0.4, min val loss = 0.0827
    600,0.2,300,0.4, min val loss = 0.1028
    650,0.2,200,0.4, min val loss = 0.1147
    550,0.2,200,0.4, min val loss = 0.2011
    300,0.3,100,0.4, min val loss = 0.1148

One layer: (neurons, dropout)
    500,0.4, min val loss = 0.1286
    400,0.4, min val loss = 0.1015 *
    300,0.4, min val loss = 0.2211
    350,0.4, min val loss = 0.1104
    400,0.3, min val loss = 0.1791
    400,0.5, min val loss = 0.1466
    200,0.5, min val loss = 0.1537
    200,0.3, min val loss = 0.1639
    600,0.3, min val loss = 0.1890
    600,0.5, min val loss = 0.1069
    600,0.6, min val loss = 0.1433
    600,0.4, min val loss = 0.1724
    100,0.1, min val loss = 0.2482
    100,0.4, min val loss = 0.1644
    
Learning rate changes with best two layer model:
    LR = 0.001, good, plateau after ~40 epochs
    LR = 0.0001, too slow, still learning after 100 epochs
    LR = 0.01, too fast, increases after ~15 epochs
    
500 Features:

Two Layers (neurons, dropout, neurons, dropout):
    250,0.3,100,0.3, min val loss =  0.1873
    300,0.3,100,0.3, min val loss = 0.2152
    200,0.3,100,0.3, min val loss = 0.2254
    250,0.2,100,0.3, min val loss = 0.1289
    250,0.1,100,0.3, min val loss = 0.1330
    250,0.4,100,0.3, min val loss = 0.1343
    250,0.2,50,0.3, min val loss = 0.2553
    250,0.2,50,0.1, min val loss = 0.1583
    250,0.2,50,0.2, min val loss = 0.1385
    250,0.2,150,0.2, min val loss = 0.1076 *
    250,0.2,150,0.3, min val loss = 0.2121
    250,0.2,150,0.1, min val loss = 0.1214
    
One Layer (neurons, dropout):
    250,0.4, min val loss = 0.2205
    100,0.3, min val loss = 0.3790
    300,0.4, min val loss = 0.1428
    350,0.4, min val loss = 0.2063
    300,0.3, min val loss = 0.1399 *
    300,0.2, min val loss = 0.1659
    
1500 Features:
    
Two Layers (neurons, dropout, neurons, dropout):
    700,0.3,200,0.3, min val loss = 0.2195
    600,0.3,200,0.3, min val loss = 0.1352
    500,0.3,200,0.3, min val loss = 0.2099
    600,0.4,200,0.3, min val loss = 0.2345
    600,0.2,200,0.3, min val loss = 0.1275
    600,0.1,200,0.3, min val loss = 0.1625
    600,0.2,250,0.3, min val loss = 0.1016 *
    600,0.2,300,0.3, min val loss = 0.1652
    600,0.2,250,0.4, min val loss = 0.1651
    600,0.2,250,0.2, min val loss = 0.1251
    
Best Model with Quantile Transformer: min val loss = 0.3663
"""

## 8 Evaluation metrics tbd
with PdfPages('Learning rate optimization.pdf') as pdf:
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    fig.suptitle("Learning Rate Optimization")
    model = models.Sequential([layers.Input(shape=(gexpTrain_scaled.shape[1],)),
                               layers.Dense(600, activation='relu'),
                               layers.Dropout(0.3),
                               layers.Dense(200, activation='relu'),
                               layers.Dropout(0.2),
                               # Binary classification = sigmoid, multiple class classification = softmax
                               layers.Dense(1, activation='sigmoid')
                               ])

    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    ## 9. Training the model
    history = model.fit(
        gexpTrain_scaled, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        verbose=2,
        #callbacks = [learning_rate],
        #class_weight = class_weights
    )
    ax[0].plot(history.history['loss'], label='train_loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Learning Rate: 0.01')
    ax[0].legend()
    
    model = models.Sequential([layers.Input(shape=(gexpTrain_scaled.shape[1],)),
                               layers.Dense(600, activation='relu'),
                               layers.Dropout(0.3),
                               layers.Dense(200, activation='relu'),
                               layers.Dropout(0.2),
                               # Binary classification = sigmoid, multiple class classification = softmax
                               layers.Dense(1, activation='sigmoid')
                               ])

    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    ## 9. Training the model
    history = model.fit(
        gexpTrain_scaled, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        verbose=2,
        #callbacks = [learning_rate],
        #class_weight = class_weights
    )
    ax[1].plot(history.history['loss'], label='train_loss')
    ax[1].plot(history.history['val_loss'], label='val_loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Learning Rate: 0.001')
    ax[1].legend()
    
    model = models.Sequential([layers.Input(shape=(gexpTrain_scaled.shape[1],)),
                               layers.Dense(600, activation='relu'),
                               layers.Dropout(0.3),
                               layers.Dense(200, activation='relu'),
                               layers.Dropout(0.2),
                               # Binary classification = sigmoid, multiple class classification = softmax
                               layers.Dense(1, activation='sigmoid')
                               ])

    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    ## 9. Training the model
    history = model.fit(
        gexpTrain_scaled, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        verbose=2,
        #callbacks = [learning_rate],
        #class_weight = class_weights
    )
    ax[2].plot(history.history['loss'], label='train_loss')
    ax[2].plot(history.history['val_loss'], label='val_loss')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Loss')
    ax[2].set_title('Learning Rate: 0.0001')
    ax[2].legend()
        
    #if 'accuracy' in history.history:
        #ax[1].plot(history.history['accuracy'], label='train_acc')
        #ax[1].plot(history.history['val_accuracy'], label='val_acc')
        #ax[1].set_xlabel('Epoch')
        #ax[1].set_ylabel('Accuracy')    
        #ax[1].legend()
    pdf.savefig()
    plt.close()

"""
Fix overfitting: too many layers or neurons, ***trying to predict too many classes****,
increase drop out, ***add validation data to training ****
np.concatenate for combining data sets, combine after going top1000
change to binary classifier (in countMe change numbers)
"""
    



