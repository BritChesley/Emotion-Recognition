#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
import datetime


# In[ ]:


from Model import Block1, Block2, Block3

# defining hyper-parameters
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.1
#TRAIN_DATA_PATH = "./train_set_2"



# In[2]:

X_train = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\X_train.npy")
y_train = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\y_train.npy")
X_val = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\X_val.npy")
y_val = np.load(r"C:\Users\bches\Classes\Spring_2021\Pattern_Recognition\Project\datasets\FER2013\train\y_val.npy")

#X_train = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/X_train.npy")
#y_train = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/y_train.npy")
#X_val = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/X_val.npy")
#y_val = np.load("/blue/wu/bchesley97/PR/datasets/FER2013/train/y_val.npy")


# In[ ]:





# In[3]:


X_train.shape, y_train.shape, X_val.shape, y_val.shape, np.unique(y_train, return_counts=True), np.unique(y_val, return_counts=True)


# In[4]:



#skf = StratifiedKFold(n_splits=2)
#for train_ind, test_ind in skf.split(X_train, y_train):
##    X_train, X_val = X_train_all[train_ind], X_train_all[test_ind]
#    y_train, y_val = y_train_all[train_ind], y_train_all[test_ind]

#X_train.shape, y_train.shape, X_val.shape, y_val.shape, np.unique(y_train, return_counts=True), np.unique(y_val, return_counts=True)


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


transform = transforms.Compose(
    [transforms.ToPILImage(), #added to do bottom transforms
   #  transforms.CenterCrop(480),
   #  transforms.Resize(224),
   #  transforms.Grayscale(3),
     transforms.RandomRotation(2),
     transforms.ToTensor(),
     transforms.Normalize(mean=train_mean, std=train_std)])


     
val_transform = transforms.Compose(
    [transforms.ToPILImage(), #added to do bottom transforms
    # transforms.CenterCrop(480),
    # transforms.Resize(224),
    # transforms.Grayscale(3),
     #transforms.RandomRotation(2),
     transforms.ToTensor(),
     transforms.Normalize(mean=train_mean, std=train_std)])


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
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            #print("input: ", imgs)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            _, pred1 = torch.max(outputs, dim=1)
            #print("_", _)
            #print("\n")
            #print("outputs: ", outputs)
            #print("predictions: ", pred1)
            #print("labels: ", labels)
            total_train += labels.shape[0]
            correct_train += (pred1==labels).sum()
            #print("Correct_train", correct_train)
            #print("labels.shape", labels.shape[0])
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
                val_errors.append(val_loss)
        #print("Validation accuracy: ", 100*correct/total)

        #if epoch == 1 or epoch % 10 == 0:
        print('{} Epoch {}, Training loss {}, Training accuracy {} Validation accuracy {}'.format(datetime.datetime.now(), epoch, float(loss_train), float(100*correct_train/total_train), float(100*correct/total)))


# In[44]:


train_dataset = my_dataset(X_train_t, y_train_t, transform=transform)

val_dataset = my_dataset(X_val_t, y_val_t, transform=val_transform)


# In[45]:


#taken from Haotians Lenet code, cite this if submitting it
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.orthogonal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)


# In[46]:


batch_size = BATCH_SIZE
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

block1 = Block1()
block2 = Block2()
block3 = Block3()
net = nn.Sequential(block1, block2, block3)
net.use_cuda = torch.cuda.is_available()

criterion = nn.CrossEntropyLoss()  # we want to use NLLLoss over here
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=0.1)
# In[47]:


net.apply(init_weights) #initialize weights to have orthogonal projection


# In[48]:
experiment_name = str(sys.argv[2])

training_loop(
    n_epochs = EPOCHS,
    optimizer = optimizer,
    model = net,
    loss_fn = criterion,
    train_loader = train_loader,
    val_loader = val_loader)


np.save(experiment_name + '_train_loss.npy', train_errors)
np.save(experiment_name + "_val_loss.npy", val_errors)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
