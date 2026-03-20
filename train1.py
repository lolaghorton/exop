import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import random_split

from load_dataset1 import LightCurveDataset
from cnn1 import smallerCNN


#configuration
BATCH_SIZE = 32
EPOCHS = 50

train_dir = "processed_lcs/train"
val_dir = "processed_lcs/validation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smallerCNN(input_length=1000).to(device)

criterion = nn.BCELoss()   #sigmoid is inside model
optimizer = optim.AdamW(model.parameters())


#class balance check
pos = len(glob.glob(f"{train_dir}/positive/CP/*.npy")) + len(glob.glob(f"{train_dir}/positive/KP/*.npy"))
neg = len(glob.glob(f"{train_dir}/negative/FP/*.npy")) + len(glob.glob(f"{train_dir}/negative/noisy/*.npy"))
print(f"Positives: {pos}, Negatives: {neg}, Pos Fraction: {pos/(pos+neg):.3f}")


#datasets 
train_dataset = LightCurveDataset(train_dir)
val_dataset = LightCurveDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# logging
metrics_log = [] 

#training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    #for X, y in train_loader:
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        
        #prints i like to see
        if epoch == 0 and batch_idx == 0:
            print("Preds:", preds[:10].detach().cpu().view(-1))
            print("Truth:", y[:10].detach().cpu().view(-1))
        
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        

    avg_train_loss = running_loss / len(train_loader)


    #evaluation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)
            loss = criterion(preds, y)
            val_loss += loss.item()
			
			#threshold at 0.5
            predicted = (preds > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    
    #log metrics
    metrics_log.append((epoch+1, avg_train_loss, avg_val_loss, val_acc))
    
    #print some stats
    print(f"Epoch {epoch+1}/{EPOCHS} | ", f"Train Loss: {avg_train_loss:.4f} | ", f"Val Loss: {avg_val_loss:.4f} | ", f"Val Acc: {val_acc:.4f}")


#save the metrics to a txt
with open("training_metrics.txt", "w") as f:
    f.write("epoch\ttrain_loss\tval_loss\tval_acc\n")
    for e, tl, vl, acc in metrics_log:
        f.write(f"{e}\t{tl}\t{vl}\t{acc}\n")


#save predictions vs truth (use full validation set, not just the training ones)
model.eval()
pred_list = []
truth_list = []

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(device)
        preds = model(X)

        preds_np = preds.cpu().numpy().flatten()
        y_np = y.numpy().flatten()

        for p, t in zip(preds_np, y_np):
            pred_list.append(p)
            truth_list.append(t)

#save preds v truth to txt
with open("vel_preds.txt", "w") as f:
	f.write("pred\ttruth\n")
	for p, t in zip(pred_list, truth_list):
		f.write(f"{p}\t{t}\n")


#save the model now trained
torch.save(model.state_dict(), "mar12_smaller1.pt")





#look into these params
'''
- metric funcs to record during training 
- hyperparameter configuration 
- system config
- logging config - olm default is BinaryAccuracy() and BinaryAUROC()
batch size - goes thru 32 lcs before updating internal parameters rather than updating for every light curve 
'''
