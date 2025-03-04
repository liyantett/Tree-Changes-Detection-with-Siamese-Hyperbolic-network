#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
import torchvision
from net.LS_1_Conv2d import S_Conv2d
from torch.hub import load_state_dict_from_url

class ConvNet(nn.Module):
    def __init__(self, BATCH_SIZE):
        super(ConvNet, self).__init__()
        
        self.conv11 = S_Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv12 = S_Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv21 = S_Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv22 = S_Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.conv31 = S_Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv32 = S_Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)       
        self.conv33 = S_Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        self.conv41 = S_Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) 
        self.conv42 = S_Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)       
        self.conv43 = S_Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        
        self.conv51 = S_Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)            
        self.conv52 = S_Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)           
        self.conv53 = S_Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        
        self.avgpool = nn.AvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2),
        )
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
      
    
    def forward(self, x):
        
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x) 
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv21(x) 
        x = self.relu(x)
        x = self.conv22(x) 
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv41(x) 
        x = self.relu(x)
        x = self.conv42(x) 
        x = self.relu(x)
        x = self.conv43(x)  
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv51(x) 
        x = self.relu(x)
        x = self.conv52(x) 
        x = self.relu(x)
        x = self.conv53(x) 
        x = self.relu(x)
        x = self.maxpool(x)
        #x = self.avgpool(x)
        #x = x.reshape(x.size(0), -1)
        #x = self.classifier(x)

        return x
        
def get_ric_vgg(BATCH_SIZE):

    model = ConvNet(BATCH_SIZE)
   
    state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
    model.load_state_dict(state_dict)
    return model
    

