import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from cnn1 import decentCNN
from load_dataset1 import LightCurveDataset

#this is for old old one, i keep just in case need to reconnect with simple stuff when lost in the sauce
''' 
model = simpleCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_dataset = lcDataset("processed_lcs", label_dict)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for X, y in train_loader:
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss = {loss.item():.4f}")
'''


#updated training for decentCNN model, doesnt include a validation section tbh, i need to get on that



#configuration
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
train_dir = "processed_lcs/train"
val_dir = "processed_lcs/validation"

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get dataset and use DataLoader
train_dataset = LightCurveDataset(train_dir)
val_dataset = LightCurveDataset(val_dir)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


#model - loss - optimizer
model = decentCNN(input_length=1000).to(device)
criterion = nn.BCELoss()   #sigmoid is inside model
optimizer = optim.Adam(model.parameters(), lr=LR)


#training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)


    #validation
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

            predicted = (preds > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    
    #print some stats
    print(f"Epoch {epoch+1}/{EPOCHS} | ", f"Train Loss: {avg_train_loss:.4f} | ", f"Val Loss: {avg_val_loss:.4f} | ", f"Val Acc: {val_acc:.4f}")


#save the model now trained
torch.save(model.state_dict(), "decentCNN.pt")

