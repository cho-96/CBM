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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
# %%
device = "cpu"
MT = pd.read_csv('D:/Users/Cho/Projects/CBM/data/MT_label.csv')
scaler = StandardScaler()
X, y = scaler.fit_transform(np.array(MT.iloc[:,:-1])), np.array(MT.iloc[:,-1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1,3)
y_train = y_train.reshape(-1,1)-1
X_test = X_test.reshape(-1,3)
y_test = y_test.reshape(-1,1)-1

#%%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
learning_rate = 0.001
batch = 100 

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor(X_train)
        self.y_data = torch.FloatTensor(y_train)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y

class CustomDataset2(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor(X_test)
        self.y_data = torch.FloatTensor(y_test)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y

train_dataset = CustomDataset()
test_dataset = CustomDataset2()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch
)

class MLP_classifier(nn.Module):
    def __init__(self,input_size,output_size, hidden):
        super(MLP_classifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, 2*hidden)
        self.fc3 = nn.Linear(2*hidden, output_size)


    def forward(self, x):

        x_1 = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        x_3 = F.relu(self.fc3(x_2))

        return x_3
    
#%%
X, y = next(iter(train_loader))
print(X)
print(torch.Tensor(y).to(torch.int64))
#%%
input_size = 3 
output_size = 5 
hidden = 500

model = MLP_classifier(input_size, output_size,hidden).to(device)
optimizer = optim.Adam(
            model.parameters(), lr=0.0001, eps=1e-10)
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

n_epoch = 200

for epoch in range(n_epoch):

    train_loss_total = 0
    test_loss_total = 0 
    for i, (input, label) in enumerate(train_loader):
        input = input.to(device)
        label = label.to(device).to(torch.int64)
        result = model(input)
        train_loss = criterion(result, label.squeeze(dim=-1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_total+=train_loss.item()

    for i, (input, label) in enumerate(test_loader):
        input = input.to(device)
        label = label.to(device).to(torch.int64)
        result = model(input)
        test_loss = criterion(result, label.squeeze(dim=-1))
        test_loss_total+=test_loss.item()*input.shape[0]
    
    test_loss_avg = test_loss_total/len(test_loader.dataset)

    if epoch%10 ==0:
        print(f'Epoch: {epoch+1:02} | Test Loss: {test_loss_avg:.4f}')
# %%
y_pred = torch.argmax(model(torch.FloatTensor(X_test).to(device)), dim=1)
y_true = y_test.reshape(-1)
# %%
y_pred = np.array(y_pred)
y_true = np.array(y_true)
# %%
y_pred
# %%
print(recall_score(y_true, y_pred, average='macro'))
print(precision_score(y_true, y_pred, average='macro'))
print(accuracy_score(y_true, y_pred))
# %%
