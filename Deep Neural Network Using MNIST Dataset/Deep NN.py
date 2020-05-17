'''
This is a demonstration of building a deep neural Network using Pytorch . This kernel can be used to generate 
a submission file for digit recognizer challenge on Kaggle

This is the first Notebook in the series of Pytorch-for-Everyone , Tutorial for this notebook is under preparation
and will soon be updated
'''


import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#importing the dataset

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
ss = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

#CONFIG
num_epoch = 10
learning_rate = 0.001
batch_size = 100


# Building a  Custom dataset 
class MNIST_dataset(torch.utils.data.Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,ids):
        
        img_as_np = np.asarray(self.data.iloc[ids,1:]).reshape(28,28).astype('uint8') 
        
        y = self.data.iloc[ids,0]
        
        if self.transform is not None:
            x = self.transform(img_as_np)
            
        return (x,y)

#Initializing The Dataloader
transform = transforms.Compose([transforms.ToTensor()]) 
train_set = MNIST_dataset(train,transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size)


# BUILDING Model
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.L1 = nn.Linear(784,128)
        self.L2 = nn.Linear(128,10)
        
    def forward(self,x):
        x = x.view(-1, 1*28*28)
        out = self.L1(x)
        out = F.relu(out)
        out = self.L2(out)
        
        return out

#Initiliazing the model
model = Mymodel()

#Defining Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training LOOP
total_step = len(train_loader)
for epoch in range(num_epoch):
    for i,data in enumerate(train_loader):
        images,labels = data
        
        optimizer.zero_grad()
        
        pred = model(images)
        
        loss = criterion(pred,labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epoch, i+1, total_step, loss.item()))


# Making Predictions on the Test Set for submission to Kaggle
with torch.no_grad():
    test_pred = []
    for i in range(len(test)):
        image = np.asarray(test.iloc[i,:]).reshape(28,28).astype('uint8')
        image = torch.from_numpy(image).float()
        
        output = model(image)
        
        _, predicted = torch.max(output.data, 1)
        
        test_pred.append(predicted.numpy())

test_pred = [val[0] for val in test_pred]

#Creating Submission File
ss['Label'] = test_pred
ss.to_csv('submission.csv',index=False)