##########################################################
## BME 598:  in_class_activity_12_2.py                  ##
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


## 1. Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, random

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score
)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


# For reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


## 2. Load dataset
data = load_breast_cancer(as_frame=True)

# pandas DataFrame of features
X = data.data

# 0 = malignant? (check below), but we'll treat as positive class = 1
y = data.target

# Confirm mapping
print("Target names:", data.target_names)  # usually ['malignant' 'benign'] or similar
print("Class distribution:\n", y.value_counts())


## 3. Train/test split (stratify to keep class ratios)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)


## 4. Preprocess: standard scaling (important for NN on tabular features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


## 5. Build a small dense network for tabular classification
# Using Sequential model
# 1. Input layer 30 nodes
# 2. Dense layer 20 nodes, activation relu
# 3. Dropout 0.3
# 4. Dense layer 20 nodes, activation relu
# 5. Dropout 0.3
# 6. Dense layer 10 nodes, activation relu
model = models.Sequential([layers.Input(shape=(X_train_scaled.shape[1],)),
                           layers.Dense(20, activation='relu'),
                           layers.Dropout(0.3),
                           layers.Dense(10, activation='relu'),
                           layers.Dropout(0.2),
                           # Binary classification = sigmoid, multiple class classification = softmax
                           layers.Dense(1, activation='sigmoid')
                           ])

model.summary()

## 6. Compile
model.compile(
    optimizer=optimizers.SGD(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


## 7. Training the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    verbose=2
)

## 8. Evaluate on test set: metrics important in biomedical context
y_proba = model.predict(X_test_scaled).ravel()
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)   # sensitivity / recall
auc = roc_auc_score(y_test, y_proba)

print(f"Test accuracy: {acc:.4f}")
print(f"Test precision: {prec:.4f}")
print(f"Test recall (sensitivity): {recall:.4f}")
print(f"Test ROC AUC: {auc:.4f}")

# Confusion matrix: [ [TN, FP], [FN, TP] ]
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Confusion matrix:\n", cm)
print(f"Specificity: {specificity:.4f}")

## 9. Plot ROC and training curves
with PdfPages('training_curves.pdf') as pdf:
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(history.history['loss'], label='train_loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
        
    if 'accuracy' in history.history:
        ax[1].plot(history.history['accuracy'], label='train_acc')
        ax[1].plot(history.history['val_accuracy'], label='val_acc')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')    
        ax[1].legend()
    pdf.savefig()
    plt.close()
    
with PdfPages('ROC_curve.pdf') as pdf:
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend()
    pdf.savefig()
    plt.close()


## 10. Save model in keras format
model.save('breast_cancer_nn_savedmodel.keras')
