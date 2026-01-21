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
        
        #remove dependence on exact time length i think (i orginally hardcoded 1000 points)
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
will prob need to use in training
criterion = nn.BCEWithLogitsLoss()
outputs = model(x)        #shape (batch, 1)
loss = criterion(outputs.squeeze(), labels.float())

for inference
prob = torch.sigmoid(output)
'''





class decentCNN(nn.Module):
    ''' 1D convolutional neural network model for light curve dataset made of .npy
        Mimics layout described in figure 3 of olm 2021 '''
    
    def __init__(self, input_length: int = 1000):
        super().__init__() #what does super do?
        self.input_length = input_length #input len of lc, ive hardcoded 1000 points


        #CONVOLUTION BLOCKS - based on olm2021
        #block0 (no BN, no dropout, pooling)
        self.conv0 = nn.Conv1d(1, 8, kernel_size=3)
        self.pool0 = nn.MaxPool1d(2)

        #block1 (BN, dropout, pooling)
        self.conv1 = nn.Conv1d(8, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(8)
        self.drop1 = nn.Dropout1d(0.1)
        self.pool1 = nn.MaxPool1d(2)

        #block2 (BN, dropout, pooling)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(16)
        self.drop2 = nn.Dropout1d(0.1)
        self.pool2 = nn.MaxPool1d(2)

        #block3 (BN, dropout, pooling)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout1d(0.1)
        self.pool3 = nn.MaxPool1d(2)

        #block4 (BN, dropout, pooling)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout1d(0.1)
        self.pool4 = nn.MaxPool1d(2)

        #block5 (BN, dropout, pooling)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(128)
        self.drop5 = nn.Dropout1d(0.1)
        self.pool5 = nn.MaxPool1d(2)

        #block6 (BN, dropout, no pooling)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(128)
        self.drop6 = nn.Dropout1d(0.1)

        #block7 (BN, dropout, no pooling)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=3)
        self.bn7 = nn.BatchNorm1d(128)
        self.drop7 = nn.Dropout1d(0.1)

        #block8 (BN, standard dropout (not spatial now), no pooling) 
        self.conv8 = nn.Conv1d(128, 20, kernel_size=3)
        self.bn8 = nn.BatchNorm1d(20)
        self.drop8 = nn.Dropout(0.1)


        #DENSE BLOCKS 
        #block9 (no BN, no dropout, no pooling)
        self.conv9 = nn.Conv1d(20, 20, kernel_size=1) 
        
        #block10 (no BN, no dropout, no pooling)
        self.conv10 = nn.Conv1d(20, 20, kernel_size=1)


        #binary output
        self.out_conv = nn.Conv1d(20, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()



    def forward(self, x):
        #x ---> (batch, 1, input_length=1000)

        x = self.pool0(self.leakyrelu(self.conv0(x)))

        x = self.pool1(self.drop1(self.bn1(self.leakyrelu(self.conv1(x)))))
        x = self.pool2(self.drop2(self.bn2(self.leakyrelu(self.conv2(x)))))
        x = self.pool3(self.drop3(self.bn3(self.leakyrelu(self.conv3(x)))))
        x = self.pool4(self.drop4(self.bn4(self.leakyrelu(self.conv4(x)))))
        x = self.pool5(self.drop5(self.bn5(self.leakyrelu(self.conv5(x)))))

        x = self.drop6(self.bn6(self.leakyrelu(self.conv6(x))))
        x = self.drop7(self.bn7(self.leakyrelu(self.conv7(x))))

        x = self.drop8(self.bn8(self.leakyrelu(self.conv8(x))))

        x = self.leakyrelu(self.conv9(x)))
        x = self.leakyrelu(self.conv10(x)))

        x = self.out_conv(x)
        x = self.sigmoid(x)

        #flatten to (batch,)
        return x.view(x.size(0))

