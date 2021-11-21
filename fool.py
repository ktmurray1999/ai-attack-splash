# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:52:27 2021

@author: murra
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from train import ResNet34

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FoolSet(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,):
        'Initialization'
        pass

  def __len__(self):
        'Denotes the total number of samples'
        return 1

  def __getitem__(self, index):
        'Generates one sample of data'
        X = image.imread('./101_ObjectCategories/stop_sign/image_0061.jpg')
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

fool_loader = torch.utils.data.DataLoader(dataset=FoolSet(),
                                          batch_size=1,
                                          shuffle=False, num_workers=0)

class Clip(nn.Module):
    def __init__(self):
        super(Clip, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input):
        output = torch.minimum(self.relu(input),torch.tensor(255))
        return output

class DreamMap(nn.Module):
    def __init__(self):
        super(DreamMap, self).__init__()
        self.noise = torch.distributions.normal.Normal(torch.zeros(300*300*3), torch.tensor([3.0])).sample()
        self.noise = torch.reshape(self.noise, (1,3,300,300))
        self.drop = nn.Dropout(p=0.5)
        self.clip = Clip()
        
        X = image.imread('./101_ObjectCategories/stop_sign/image_0061.jpg')
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
        X = torch.unsqueeze(X, 0)
        X = torch.permute(X, (0, 3, 1, 2)).float()

        self.image = nn.Parameter(X + self.noise)

    def forward(self):
        return self.drop(self.image)

def EvalInitialImage(model):
    for k, data in enumerate(fool_loader, 0):
        plt.imshow(torch.squeeze(data[0]).detach().numpy())
        plt.show()
        inputs = torch.permute(data[0].to(device), (0, 3, 1, 2)).float()
        out = F.softmax(model(inputs), dim=-1)
    print(out)
    return out

def FoolScript(model, epochs):
    model.eval()
    deepdream = DreamMap()
    deepdream.train()
    optimizer = optim.Adam(deepdream.parameters())
    for i in range(epochs):
        optimizer.zero_grad()
        inputs = deepdream()
        out = model(inputs)
        loss = out[0,-1] - out[0,0]
        loss.backward()
        optimizer.step()
    
    probabilities = F.softmax(out, dim=-1)
    print(probabilities)
    return deepdream, probabilities

def ShowImages(deepdream):
    deepdream.eval()
    clipper = Clip()
    inputs = clipper(deepdream())
    img = torch.squeeze(inputs)
    img = torch.permute(img, (1, 2, 0))
    plt.imshow(img.detach().numpy().astype(np.intc))
    plt.show()
        
    return None

if __name__ == "__main__":
    model = ResNet34(pretrained=True).to(device)
    model.load_state_dict(torch.load('model_accurate.pt'))
    model.eval()
    out_init = EvalInitialImage(model)
    deepdream, out_final = FoolScript(model, 750)
    ShowImages(deepdream)
