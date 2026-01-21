import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)

        self.pool = nn.MaxPool1d(kernel_size=2)

        #input is 1000 values then after 3 poolings get 1000/8 = 125
        self.fc1 = nn.Linear(64 * 125, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  #binary classification

        return x

class midCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequantial(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(), 
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.MaxPool1d(2), 
        )
        
        #remove dependence on exact time length (i orginally hardcoded 1000 points)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequantial(
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(64, 1) #logits
        )
        
    def forward(self, x):
        #x shape -> (batch, 1, 1000)
        x = self.features(x)
        x = self.global_pool(x) #(batch, 64, 1)
        x = x.squeeze(-1) #(batch, 64)
        x = self.classifier(x)
        return x
'''
for midCNN
use in training
criterion = nn.BCEWithLogitsLoss()

outputs = model(x)        # shape (batch, 1)
loss = criterion(outputs.squeeze(), labels.float())

for inference
prob = torch.sigmoid(output)
'''




class decentCNN(self):
    return True
        
