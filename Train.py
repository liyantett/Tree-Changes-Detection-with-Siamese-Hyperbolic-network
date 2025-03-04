import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from TreeDataset import VoxTree as VoxTree
from torchvision.transforms import ToTensor, Resize, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize,RandomRotation,Normalize
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import numpy
import os
import cv2
from scipy.spatial import distance_matrix
from hyptorch.pmath import dist_matrix
import hyptorch.nn as hypnn
from tqdm import tqdm
from net.resnet import resnet18
import argparse


def set_seed(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)


parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--balldims', type=int,default='8',help='Ball dimensions')
parser.add_argument('--hyp_c', type=float,default=0.7,help='hyperbolic c, 0 enables sphere mode')
parser.add_argument('--clip_r', type=float,default=2.3,help='feature clipping radius')
parser.add_argument('--ep', type=int,default=80,help='training epoch')


bs=parser.bs
balldims=parser.balldims
ep = parser.ep
hyp_c =parser.hyp_c 
clip_r =parser.clip_r  


""
############################         Siamese    hyperbolic network          ###############################

""
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        pretrained=True
        self.resnet = resnet18()
        del self.resnet.fc 
        flat_shape = 512 *7*7
        self.fully_connect= torch.nn.Linear(flat_shape, balldims)
     

        self.tp = hypnn.ToPoincare(
            c=hyp_c,
            ball_dim=emb,
            riemannian=False,
            clip_r=clip_r ,)        
        self.mlr = hypnn.HyperbolicMLR(ball_dim=balldims, n_classes=2, c=hyp_c)
          
    def forward(self, input1, input2):
        f,f1 = self.resnet(input1)
        f,f2 = self.resnet(input2)  
        x1 = torch.flatten(f1, 1)
        x2 = torch.flatten(f2, 1)
        x = torch.abs(x1 - x2)
        x = self.fully_connect(x)
        x = self.tp(x)
       
        return x,F.log_softmax(self.mlr(x), dim=-1)


train_set = VoxTree(2,'train', 100, 1, jittering=True)
train_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=bs, shuffle=True)

counter = []
loss_history = [] 
iteration_number= 0
tloss= 0

net = SiameseNetwork().cuda()
device = torch.device("cuda")


""
############################         Train          ###############################

""

for epoch in range(0,ep): 
    #SGD
    lr_dec=5
    optimizer = optim.SGD([{'params' : net.resnet.parameters(), 'lr':0.0000001},{'params' : net.fully_connect.parameters(), 'lr':0.001*(0.1**(epoch//lr_dec))},{'params' : net.tp.parameters(), 'lr':0.001*(0.1**(epoch//lr_dec))},{'params' : net.mlr.parameters(), 'lr':0.001*(0.1**(epoch//lr_dec))}],momentum=0.5)
    net.train()
    for i, data in enumerate(train_data_loader,0):
        img0, img1 , label,name0,name1 = data
        img0, img1 , label = img0.to(device), img1.to(device), label.to(device).long()
        optimizer.zero_grad()
        feature, output = net(img0,img1)        
        label=torch.squeeze(label,1).long()
        loss = F.nll_loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tloss= tloss+ loss
      
    if epoch == ep :
        print("Epoch number {}\n Current loss {}\n".format(epoch,tloss))
        net.eval()
        tloss= 0
        test_set = VoxTree(2,'test', 100, 2)
        test_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=int(bs))

        targets_all=[]
        predict_all=[]
        feature_all=[]
        output_all=[]
        for i, data in enumerate(test_data_loader,0):
                img0, img1 , label,name0,name1 = data 
                img0, img1 , label = img0.to(device), img1.to(device), label.to(device).long()
       	        feature, output = net(img0,img1)
                label=torch.squeeze(label,1).long()
       
                loss = F.nll_loss(output, label)
                predict= output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                traget=label.cpu().detach().numpy()
                feature=feature.cpu().detach().numpy()
                output=output.cpu().detach().numpy()
                targets_all=numpy.append(targets_all,traget)
                predict_all=numpy.append(predict_all,predict.cpu())

                if i==0:
                    feature_all=feature
                    output_all=output
                else:
                    feature_all=numpy.concatenate((feature_all,feature)) 
                    output_all=numpy.concatenate((output_all,output)) 

               
        f1score= f1_score(targets_all, predict_all,average='binary')
        f1score_macro=f1_score(targets_all, predict_all,average='macro')
        acc=accuracy_score(targets_all,predict_all)
        print('ACC:',acc,'F1:',f1score,f1score_macro)
        print(feature_all.shape,output_all.shape,targets_all.shape)
        
        
      





