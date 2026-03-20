#overfitting training script

# ---> this is to ensure even with meh results, that the model can in fact learn, to the point of memorization, then we know something else is wrong and the model structure is sane

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


#some set up stuff
train_dir = "processed_lcs/train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smallerCNN(input_length=1000).to(device)
criterion = nn.BCELoss()   
optimizer = optim.AdamW(model.parameters())


#class balance check
pos = len(glob.glob(f"{train_dir}/positive/CP/*.npy")) + len(glob.glob(f"{train_dir}/positive/KP/*.npy"))
neg = len(glob.glob(f"{train_dir}/negative/FP/*.npy")) + len(glob.glob(f"{train_dir}/negative/noisy/*.npy"))
print(f"Positives: {pos}, Negatives: {neg}, Pos Fraction: {pos/(pos+neg):.3f}")


#dataset
full_dataset = LightCurveDataset(train_dir)
train_size = int(0.8 * len(full_dataset)) 
val_size = len(full_dataset) - train_size #just using parts of the train set for this
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


#mini set to test overfitting
tiny_indices = list(range(10))
tiny_dataset = Subset(train_dataset, tiny_indices)
tiny_loader = DataLoader(tiny_dataset, batch_size=5, shuffle=True) #normal batch size is like 32 for reference


#to track loss func vs epochs
epoch_losses = []

# the actual overfittingness of the training script
for epoch in range(200): #run through 200 times and calc loss
    model.train()
    running_loss = 0.0

    for X, y in tiny_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    epoch_losses.append((epoch, running_loss))

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss {running_loss:.4f}") #only print so many of these
        
#log the loss in txt file
with open("overfit_loss.txt", "w") as f:
	f.write("epoch\tloss\n")
	for epoch loss in epoch_losses:
		f.write(f"{epoch}\t{loss}\n")

#validation
model.eval()
pred_list = []
truth_list = []

with torch.no_grad():
    for X, y in tiny_loader:
        X = X.to(device)
        preds = model(X)
		
		preds_np = preds.cpu().numpy().flatten()
		y_np = y.numpy().flatten()
		
		for p, t in zip(preds_np, y_np):
			pred_list.append(p)
			truth_list.append(t)
		
        print("Preds:", preds_np.round(3))
        print("Truth:", y_np)
        break

#log the predictions vs truth in txt
with open("overfit_preds.txt", "w") as f:
	f.write("pred\ttruth\n")
	for p, t in zip(pred_list, truth_list):
		f.write(f"{p}\t{t}\n")

#save the over-trained model
torch.save(model.state_dict(), "overfit_march_17.pt")


