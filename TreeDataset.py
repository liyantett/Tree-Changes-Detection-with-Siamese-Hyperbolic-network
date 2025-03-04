# Load the datase

import torch.utils.data as data
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor, Resize, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize,RandomRotation,Normalize
from PIL import Image
import pandas as pd
import torch
import os


def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img

def load_img_mask(file_path):
	img = Image.open(file_path).convert('RGB')
	return img



TREE_LOCATION='/users/liyante1/Treedata/data/'
csv_path='/users/liyante1/Treedata/clean_remove_tree_1108training.xlsx'
TREE_LOCATIONtest='/users/liyante1/Treedata/CanadaDate/'

days= 65
bs=128

def convert_train(csv_path, subset):
    df= pd.read_excel(csv_path, sheet_name=subset)
    keys = []
    key_labels = []
    key_imgs = []
    subsets = []
    image_names=df.loc[0]

    print(image_names)

    for i in range(1,len(df)):
 
        tmp=df.loc[i]
        #print(tmp)
        basename = TREE_LOCATION+ '/' +str(int(tmp['Tree']))
        img_name=[]
        class_name=[]
        for j in range(1,days,2):
            for k in range(j+1,days,3):
                name0 = basename + '/' +str(int(image_names[j]))+'.jpg'
                name1 = basename + '/' +str(int(image_names[k]))+'.jpg'
                
                class0=tmp[j]
                class1=tmp[k]
                img_name = np.append(name0, name1)
                class_name=np.append(class0, class1)
                #print(image_names[j])
                #print(class_name)
                key_labels.append(class_name)
                key_imgs.append(img_name)
                #print(int(tmp['Tree']),image_names[j],image_names[k],class0,class1)
        
    return key_imgs,key_labels

    






def convert_test(csv_path, subset):
    df= pd.read_excel(csv_path, sheet_name=subset)
    keys = []
    key_labels = []
    key_imgs = []
    subsets = []
    image_names=df.loc[0]
    print(image_names)
    for i in range(1,len(df)):
        tmp=df.loc[i]
        basename = TREE_LOCATION+ '/' +str(int(tmp['Tree']))
    
        img_name=[]
        class_name=[]
        for j in range(1,days):
            for k in range(j,days,2):
                #print(int(image_names[j]))
                name0 = basename + '/' +str(int(image_names[j]))+'.jpg'
                name1 = basename + '/' +str(int(image_names[k]))+'.jpg'
                class0=tmp[j]
                class1=tmp[k]
                img_name = np.append(name0, name1)
                class_name=np.append(class0, class1)
               
                key_labels.append(class_name)
                key_imgs.append(img_name)
                #print(int(tmp['Tree']),image_names[j],image_names[k],class0,class1)

    return key_imgs,key_labels     






def convert_test_extra(csv_path, subset):
    df= pd.read_excel(csv_path, sheet_name=subset)
    keys = []
    key_labels = []
    key_imgs = []
    subsets = []
    image_names=df.loc[0]
    print(image_names)
    for i in range(1,len(df)):
        #print(i)
        #print(df.loc[i])
        
        tmp=df.loc[i]
        basename = TREE_LOCATIONtest+ '/' +str(int(tmp['Tree']))
    
        img_name=[]
        class_name=[]
        for j in range(1,7):
            for k in range(j,7,2):
                #print(int(image_names[j]))
                name0 = basename + '/' +str(int(image_names[j]))+'.jpg'
                name1 = basename + '/' +str(int(image_names[k]))+'.jpg'
                class0=tmp[j]
                class1=tmp[k]
                img_name = np.append(name0, name1)
                class_name=np.append(class0, class1)
               
                key_labels.append(class_name)
                key_imgs.append(img_name)
                #print(int(tmp['Tree']),image_names[j],image_names[k],class0,class1)     
    return key_imgs,key_labels     






class VoxTree(data.Dataset):
	def __init__(self, num_views, subset,random_seed, dataset, additional_face=True, jittering=False):
                if subset == 'train':
                        subset='train'
                        key_imgs,key_labels=convert_train(csv_path, subset)      #csv_path_test
                        self.images = key_imgs    
                        self.labels=key_labels
                       
                if subset == 'test':
                        subset='test'
                        key_imgs,key_labels=convert_test(csv_path, subset)     #csv_path_train
                       
                        self.images = key_imgs    
                        self.labels=key_labels   
                                           
                if subset == 'extratest':
                        subset='extratest'
                        key_imgs,key_labels=convert_test_extra(csv_path, subset)     #csv_path_train
                       
                        self.images = key_imgs    
                        self.labels=key_labels
               
                self.rng =  np.random.RandomState(random_seed)   #np.random.default_rng()#
                self.num_views = num_views
                self.base_file = TREE_LOCATION+ '/%s/' 
                crop = 180
                if jittering == True:
                    precrop = crop + 5
                    crop = self.rng.randint(crop, precrop)
                    self.pose_transform = Compose([Resize((224,224)),
                                               CenterCrop(precrop), RandomCrop(crop),
                                               Resize((224,224)), ToTensor()])
                    self.transform = Compose([Resize((224,224)), ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                else:
                    precrop = crop
                    self.pose_transform = Compose([Resize((256,256)),
                                               #Pad((20,80,20,30)),
                                               CenterCrop(precrop),
                                               ToTensor()])
                    self.transform = Compose([Resize((224,224)), ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    #,Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	
	def __len__(self):
		
		return len(self.labels) - 1
	def __getitem__(self, index):
		#(other_face, _) = self.get_blw_item(self.rng.randint(self.__len__()))
		#print('++++++++++++++++++++++++++++++++++')
		return self.get_blw_item(index)
	
	def get_blw_item(self, index):
		# Load the imag
       
                               
                imgs = [0] * (self.num_views)
                labels = [0] * (self.num_views)
                imgs_name = [0] * (self.num_views)
                #print('+++++++++++++++++++++++++++++++++') 
                img_name0 = self.images[index][0]
                img_name1 = self.images[index][1]      
                imgs[0] = load_img(img_name0)
                imgs[1] = load_img(img_name1)       
                imgs[0] = self.transform(imgs[0])
                imgs[1] = self.transform(imgs[1])       
                labels[0]=self.labels[index][0]
                labels[1]=self.labels[index][1]
                        #print(img_index[i],imgs_name[i],labels[i])
                        
               
                if labels[0]==labels[1]:
                        label=1
                else:
                        label=0
                #print(labels[0],labels[1],np.array([int(labels[0]!=labels[1])],dtype=np.float32))
                return imgs[0],imgs[1], torch.from_numpy(np.array([int(labels[0]!=labels[1])],dtype=np.float32)),imgs_name[0],imgs_name[1]





