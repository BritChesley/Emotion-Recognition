#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys
import torch.linalg

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
#from torch.optim import
from torchvision import transforms
import datetime
from torch.nn.utils import parameters_to_vector

# In[12]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810*2,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )


        self.localization_2 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(24, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc_2 = nn.Sequential(
            nn.Linear(3200, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc_2[2].weight.data.zero_()
        self.fc_loc_2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


    def stn_2(self, x):
        xs = self.localization_2(x)
        #print("xs.shape ", xs.shape)
        xs = xs.view(-1, 3200)
        theta = self.fc_loc_2(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners = True)
        x = F.grid_sample(x, grid, align_corners = True)
        return x

    def forward(self,input):
        out = self.stn(input)
        out2 = self.stn_2(input)

        #print("out.shape: ", out.shape)
        #print("out2.shape: ", out2.shape)

        out = torch.cat((out, out2))
        #print("concat shape: ", out.shape)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        #print("fc layer:", out.shape )
        #out = out.view(-1, 810)
        out = out.view(-1, 810*2)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        #print(parameters_to_vector(list(self.fc1.parameters())).shape)
        #print(parameters_to_vector(list(self.fc2.parameters())).shape)

        return out


# In[13]:




X_train = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\X_train.npy") #np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/X_train.npy")
y_train = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\y_train.npy") #np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/y_train.npy")
X_val = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\X_val.npy") #np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/X_val.npy")
y_val = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\y_val.npy") #np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/y_val.npy")



#X_train = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/X_train.npy")
#y_train = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/y_train.npy")
#X_val = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/X_val.npy")
#y_val = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/y_val.npy")


# In[ ]:





# In[3]:


X_train.shape, y_train.shape, X_val.shape, y_val.shape, np.unique(y_train, return_counts=True), np.unique(y_val, return_counts=True)


# In[4]:




# In[5]:


#convert data from numpy to tensors
X_train_t = torch.tensor(X_train.tolist(), dtype=torch.float32)/255
y_train_t = torch.tensor(y_train.tolist(), dtype=torch.long)
X_val_t = torch.tensor(X_val.tolist(), dtype=torch.float32)/255
y_val_t = torch.tensor(y_val.tolist(), dtype=torch.long)


X_train_t.type(torch.float32)
y_train_t.type(torch.long)
X_val_t.type(torch.float32)
y_val_t.type(torch.long)

X_train_t.shape, X_train_t.type, y_train_t.shape


# In[6]:


#pytorch tensors require N X C X H X W
X_train_t = X_train_t.unsqueeze(1).contiguous()
X_val_t = X_val_t.unsqueeze(1).contiguous()

X_train_t.shape, X_val_t.shape, y_train_t[0].type


# In[7]:


X_train_t.min(), X_train_t.max(), y_train_t.min() #double check to make sure min is 0 and max is 1


# In[8]:


X_train_t.view(1,-1).mean(dim=1), X_train_t.view(1,-1).std(dim=1) #check mean and std deviation values


# In[9]:


train_mean = X_train_t.view(1,-1).mean(dim=1)
train_std = X_train_t.view(1,-1).std(dim=1)


# In[10]:


train_transform = torchvision.transforms.Compose([
            #torchvision.ToPILImage(), #need this to do data augmentation, only accepts PIL images
            #torchvision.transforms.Resize(48), #48 is FER2013 size
            #torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std)
])

val_transform = torchvision.transforms.Compose([
            #torchvision.ToPILImage(), #need this to do data augmentation, only accepts PIL images
            #torchvision.transforms.Resize(48), #48 is FER2013 size
            #torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std)
])

# In[11]:


#data set class definition

class my_dataset(Dataset):
    def __init__(self, X, y, transform = None):
        self.data = X
        self.target = y
        self.transform = transform

        if torch.cuda.is_available():
            print("Data placed in GPU memory")
            self.data = self.data.cuda()
            self.target = self.target.cuda()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x.cpu())

        if torch.cuda.is_available():
            return x.cuda(), y.cuda()

        return x,y

    def __len__(self):
        return len(self.data)



# In[12]:


num_classes = 7

train_errors = []
val_errors = []
#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle=True)
device = torch.device("cuda:0")
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    patience = 50 #50 epoch patience for early stopping
    lamb = 0.001
    if model.use_cuda:
        if(torch.cuda.device_count() > 1):
            print("Using data parallel for training model")
            model = nn.DataParallel(model)
        model.to(device)
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        val_loss = 0.0
        correct_train = 0
        total_train = 0
        for imgs, labels in train_loader:
            model.train()
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            #l2 = torch.cat((parameters_to_vector(model.fc1.parameters()), parameters_to_vector(model.fc2.parameters())), 0)
            #l2 = torch.norm(l2, 2)
            #print("l2 shape: ", l2)
            #loss += lamb*l2

            #print("_", _)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            _, pred1 = torch.max(outputs, dim=1)
            total_train += labels.shape[0]
            correct_train += (pred1==labels).sum()


            #print("\n")
            #print("outputs: ", outputs)
            #print("predictions: ", pred1)
            #print("labels: ", labels)

            #train_acc = 100*correct_train/total_train

        train_errors.append(loss_train)

        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for imgs1, label in val_loader:
                imgs1 = imgs1.to(device)
                outputs1 = model(imgs1)
                #print(outputs1)
                _, pred = torch.max(outputs1, dim=1)
                total += label.shape[0]
                correct += (pred == label).sum()
                val_loss += loss_fn(outputs1, label)
                l2_val = torch.cat((parameters_to_vector(model.fc1.parameters()), parameters_to_vector(model.fc2.parameters())), 0)
                val_loss += lamb * torch.norm(l2_val, 2)

                val_errors.append(val_loss)
        #print("Validation accuracy: ", 100*correct/total)

        if epoch == 1 or epoch % 10 == 0:
              print('{} Epoch {}, Training loss {}, Training accuracy {} Validation accuracy {}'.format(datetime.datetime.now(), epoch, float(loss_train), float(100*float(correct_train)/float(total_train)), float(100*float(correct)/float(total))))


# In[44]:


train_dataset = my_dataset(X_train_t, y_train_t, transform=train_transform)

val_dataset = my_dataset(X_val_t, y_val_t, transform=train_transform)


# In[45]:


#taken from Haotians Lenet code, cite this if submitting it
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.orthogonal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)


# In[46]:


batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = Deep_Emotion()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
sum(p.numel() for p in model.parameters())


# In[47]:


#model.apply(init_weights) #initialize weights to have orthogonal projection


# In[48]:
experiment_name = str(sys.argv[2])

training_loop(
    n_epochs = 1000,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    val_loader = val_loader)


np.save(experiment_name + '_train_loss.npy', train_errors)
np.save(experiment_name + "_val_loss.npy", val_errors)

torch.save(model.state_dict(), experiment_name+'.pt')
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# In[ ]:




