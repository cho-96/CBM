#%%
from cProfile import label
import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd 
import os 
import sys 
import random 
import torch.nn.functional as F
import torch.optim as optim

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"

# Input data 
input_data = np.ones(shape = (2,3,2))
input_target = np.zeros(shape = (2,3,1))

# %%
learning_rate = 0.001
batch_size = 100 

class MLP_classifier(nn.Module):
    def __init__(self,input_size,output_size, hidden):
        super(MLP_classifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden*2)
        self.fc2 = nn.Linear(hidden*2, hidden)
        self.fc3 = nn.Linear(hidden,output_size )


    def forward(self, x):

        x_1 = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        x_3 = F.relu(self.fc3(x_2))

        return x_3
#%%
input_size = 2 
output_size = 1 
hidden = 128

model = MLP_classifier(input_size, output_size,hidden).to(device)
optimizer = optim.Adam(
            model.parameters(), lr=0.0001, eps=1e-10)
criterion = nn.CrossEntropyLoss()


n_epoch = 1000 
for epoch in range(n_epoch):

    train_loss_total = 0
    test_loss_total = 0 
    for i, (input, label, _) in enumerate(dm.train_loader):

        result = model(input)
        train_loss = criterion(result,label)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_total+=train_loss.item()

    for i, (input, label, _) in enumerate(dm.test_loader):
        result = model(input)
        test_loss = criterion(result,label)

        test_loss_total+=test_loss.item()

    if test_loss_total < best_test_loss:
        best_test_loss = test_loss_total
        torch.save(model.state_dict(),
                'model_save/DNN_100_full_non_filter_1.pth')

    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss_total:.4f}')
    if epoch%10 ==0:
        print(f'Epoch: {epoch+1:02} | Test Loss: {test_loss_total:.4f}')

