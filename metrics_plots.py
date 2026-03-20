#to plot up various data gotten from training/validation and overfitting 

import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---- LOSS FUNCTIONS ----

#load in the training metrics (loss funcs n such) 
train_metrics_file = "training_metrics.txt"
val_preds_file = "val_preds.txt" 

train_data = np.loadtxt(train_metrics_file, skiprows=1)
epochs = train_data[:, 0]
train_loss = train_data[:, 1]
val_loss = train_data[:, 2]
val_acc = train_data[:, 3]

#plot the loss functions
plt.figure(figsize=(10, 5))

plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.text(1, 0.1, f"Validation Accuracy of {val_acc}"

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

'''
#load in overfitting metrics 
overfit_metrics_file = "overfit_loss.txt"
overfit_preds_file = "overfit_preds.txt" 

overfit_data = np.loadtxt(overfit_metrics_file, skiprows=1)
overfit_epochs = overfit_data[:, 0]
overfit_loss = overfit_data[:, 1]


#overfit loss
plt.figure(figsize=(10, 5))

plt.plot(overfit_epochs, overfit_loss, label="Overfit Loss", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfit Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
'''




# ---- CONFUSION MATRIX ----

#func to load predictions and truth tensor 
def load_preds(file):
    data = np.loadtxt(file, skiprows=1)
    preds = data[:, 0]
    truth = data[:, 1]
    return preds, truth


#func to make confusion matrix 
def plot_conf_matrix(preds, truth, title):
    #threshold at 0.5
    preds_binary = (preds > 0.5).astype(int)
    truth = truth.astype(int)

    cm = confusion_matrix(truth, preds_binary)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(title)
    plt.tight_layout()
    plt.show()


#validation confusion matrix (from training script)
val_preds, val_truth = load_preds(val_preds_file)
plot_conf_matrix(val_preds, val_truth, "Validation Confusion Matrix")

'''
#overfit confusion matrix 
overfit_preds, overfit_truth = load_preds(overfit_preds_file)
plot_conf_matrix(overfit_preds, overfit_truth, "Overfit Confusion Matrix")
'''

