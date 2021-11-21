# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:20:49 2021

@author: murra
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils
import numpy as np
import pretrainedmodels
from matplotlib import image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainSet(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,):
        'Initialization'
        pass

  def __len__(self):
        'Denotes the total number of samples'
        return 180

  def __getitem__(self, index):
        'Generates one sample of data'
        num = index + 1
        
        if num <= 60:
            if num < 10:
                X = image.imread('./101_ObjectCategories/soccer_ball/image_000'+str(num)+'.jpg')
            else:
                X = image.imread('./101_ObjectCategories/soccer_ball/image_00'+str(num)+'.jpg')
            y = torch.tensor([0])
        elif 60 < num <= 120:
            if num-60 < 10:
                X = image.imread('./101_ObjectCategories/dalmatian/image_000'+str(num-60)+'.jpg')
            else:
                X = image.imread('./101_ObjectCategories/dalmatian/image_00'+str(num-60)+'.jpg')
            y = torch.tensor([1])
        elif 120 < num:
            if num-120 < 10:
                X = image.imread('./101_ObjectCategories/stop_sign/image_000'+str(num-120)+'.jpg')
            else:
                X = image.imread('./101_ObjectCategories/stop_sign/image_00'+str(num-120)+'.jpg')
            y = torch.tensor([2])
        X = torch.from_numpy(X)
        if len(X.shape) < 3:
            X = torch.stack((X,X,X),-1)
        if X.shape[0] < 300:
            remainder = int((300-X.shape[0])/2)
            X = F.pad(X, (0,0,0,0,remainder,remainder), "constant", 0)
        if X.shape[0] == 299:
            X = F.pad(X, (0,0,0,0,1,0), "constant", 0)
        if X.shape[1] < 300:
            remainder = int((300-X.shape[1])/2)
            X = F.pad(X, (0,0,remainder,remainder,0,0), "constant", 0)
        if X.shape[1] == 299:
            X = F.pad(X, (0,0,1,0,0,0), "constant", 0)
        return X, y

train_loader = torch.utils.data.DataLoader(dataset=TrainSet(),
                                          batch_size=5,
                                          shuffle=True, num_workers=0)

class TestSet(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,):
        'Initialization'
        pass

  def __len__(self):
        'Denotes the total number of samples'
        return 15

  def __getitem__(self, index):
        'Generates one sample of data'
        num = index + 1
        
        if num <= 4:
            X = image.imread('./101_ObjectCategories/soccer_ball/image_006'+str(num)+'.jpg')
            y = torch.tensor([0])
        elif 4 < num <= 11:
            X = image.imread('./101_ObjectCategories/dalmatian/image_006'+str(num-4)+'.jpg')
            y = torch.tensor([1])
        elif 11 < num:
            X = image.imread('./101_ObjectCategories/stop_sign/image_006'+str(num-11)+'.jpg')
            y = torch.tensor([2])
        X = torch.from_numpy(X)
        if len(X.shape) < 3:
            X = torch.stack((X,X,X),-1)
        if X.shape[0] < 300:
            remainder = int((300-X.shape[0])/2)
            X = F.pad(X, (0,0,0,0,remainder,remainder), "constant", 0)
        if X.shape[0] == 299:
            X = F.pad(X, (0,0,0,0,1,0), "constant", 0)
        if X.shape[1] < 300:
            remainder = int((300-X.shape[1])/2)
            X = F.pad(X, (0,0,remainder,remainder,0,0), "constant", 0)
        if X.shape[1] == 299:
            X = F.pad(X, (0,0,1,0,0,0), "constant", 0)
        return X, y

test_loader = torch.utils.data.DataLoader(dataset=TestSet(),
                                          batch_size=15,
                                          shuffle=False, num_workers=0)

def Test(model):
    correct = 0
    model.eval()
    for k, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = torch.permute(inputs, (0, 3, 1, 2)).float()
        labels = torch.squeeze(labels)
        
        out = model(inputs)
        out = torch.argmax(out, dim=-1)
        compare = torch.eq(out, labels)
        correct += torch.sum(compare)
    
    return correct/len(TestSet())

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)
        
        # change the classification layer
        self.l0 = nn.Linear(512, 3)
        self.dropout = nn.Dropout2d(0.05)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0

def Train(model, epoch):
    model.train()
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    test_accuracies = []
    for i in range(epoch):
            for k, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = torch.permute(inputs, (0, 3, 1, 2)).float()
                labels = torch.squeeze(labels)
                model.train()
                
                optimizer.zero_grad()
                out = model(inputs)
                loss = objective(out, labels)
                loss.backward()
                optimizer.step()
            
            test_acc = Test(model).item()
            test_accuracies.append(test_acc)
            print(test_acc)
            if test_acc == 1.0:
                torch.save(model.state_dict(), 'model_accurate.pt')
                break
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,epoch+1), test_accuracies)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Testing accuracy across epochs')
    plt.show()
    
    return test_accuracies

if __name__ == "__main__":
    model = ResNet34(pretrained=True).to(device)
    test_accuracies = Train(model, 50)
    torch.save(model.state_dict(), 'model.pt')

    
