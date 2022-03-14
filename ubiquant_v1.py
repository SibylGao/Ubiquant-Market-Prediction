# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:16:39 2022

@author: 92108
"""
#!D:/QLDownload/ubiquant-market-prediction/python
import torch
from torch import nn as nn
from torch import tensor as tensor
import numpy as np
import gc
import torch.optim as optim
import pandas as pd

class swish(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)
    
class CATMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(3774, 32,max_norm=32,norm_type=2)
        self.linear1=nn.Linear(32,64)
        self.linear2=nn.Linear(64,32)
        self.swish=swish()
        self.linear3=nn.Linear(300,256)
        self.tanh=nn.Tanh()
        self.bn=nn.BatchNorm1d(1,64,momentum=0.01)
        self.linear4=nn.Linear(256,128)
        self.linear5=nn.Linear(96,64)
        self.linear4_1=nn.Linear(128,64)
        self.linear6=nn.Linear(64,32)
        self.linear7=nn.Linear(32,16)
        self.linear8=nn.Linear(16,1)
        self.dropout=nn.Dropout(p=0.5)
        nn.init.kaiming_normal_(self.linear2.weight.detach())
        nn.init.kaiming_normal_(self.linear3.weight.detach())

        
    def forward(self,x_1,x_2):
        x_1_v=self.embedding(x_1)
        x_1_h1=self.linear1(x_1_v)
        x_1_h1=self.swish(x_1_h1)
        x_1_h1=self.dropout(x_1_h1)
        x_1_h2=self.linear2(x_1_h1)
        x_1_g=self.swish(x_1_h2)
        x_1_g=self.dropout(x_1_h2)
        
        x_2_h1=self.linear3(x_2)
        x_2_h1=self.swish(x_2_h1)
        x_2_h1=self.dropout(x_2_h1)
        x_2_h2=self.linear4(x_2_h1)
        x_2_h2=self.swish(x_2_h2)
        x_2_h2=self.dropout(x_2_h2)
        x_2_h3=self.linear4_1(x_2_h2)
        x_2_h3=self.swish(x_2_h3)
        
        y_h1=torch.cat((x_1_g,x_2_h3),2)
        y_h2=self.linear5(y_h1)
        y_h2=self.swish(y_h2)
       # y_h2=self.bn(y_h2)
        y_h3=self.linear6(y_h2)
        y_h3=self.swish(y_h3)
        y_h4=self.linear7(y_h3)
        y_h4=self.swish(y_h4)
        y=self.linear8(y_h4)
        return y
    
class DataLoader():
    def __init__(self,df,device='cuda'):
        super().__init__()
        self.device=device
        df.pop('time_id')
        df.pop('row_id')
        self.df=df
        
    def getdata(self,batch_size=64):
        dfi=self.df.sample(n=batch_size,replace=False,axis=0)
        target=dfi.pop('target').values.reshape(len(dfi),1,1)
        input_1=dfi.pop('investment_id').values.reshape(len(dfi),1)
        input_2=dfi.values.reshape(len(dfi),1,300)
        x_1=tensor(input_1,device=self.device,dtype=torch.long)
        x_2=tensor(input_2,device=self.device,dtype=torch.float32)
        y=tensor(target,device=self.device,dtype=torch.float32)
        return x_1,x_2,y
    
def tcorr(x,y):
    x_mean=x.mean(dim=0)
    y_mean=y.mean(dim=0)
    x_d=torch.add(x,-x_mean)
    y_d=torch.add(y,-y_mean)
    corr=sum(torch.mul(x_d,y_d))/(sum(torch.mul(x_d,x_d)).sqrt()*sum(torch.mul(y_d,y_d)).sqrt())
    return corr

def loss_function(x,y):
    return nn.MSELoss()(x,y)-tcorr(x,y)+1
    
def train_epoch(model, device, train_loader, optimizer, epoch,batch_size=None):
    loss_fn = torch.nn.MSELoss()
    model.train()
    aloss=[]
    for i in range(1000):
        data=train_loader.getdata(batch_size=batch_size)
        if data==None:
            continue
        input_1,input_2,target=data
        optimizer.zero_grad()
        output = model(input_1,input_2)
        loss =  loss_function(output, target)
       # loss=-tcorr(output,target)
        loss.backward()
        optimizer.step()
        aloss.append(loss.item())
        if i % 200==199:
            print("epoch:%.0f batch:%.0f loss:%.4f"%(epoch+1,i+1,sum(aloss)/len(aloss)))
        del loss
    del aloss
            
def test_epoch(model, device, test_loader,batch_size=None):
    model.eval()
    corr=[]
    with torch.no_grad():
        for i in range(2000):
            data=test_loader.getdata(batch_size=batch_size)
            if data==None:
                continue
            input_1,input_2,target=data
            output = model(input_1,input_2)
            x=target.cpu().numpy()
            x=x.reshape(1,len(x))
            y=output.cpu().numpy()
            y=y.reshape(1,len(y))
            mul_std=x.std()*y.std()
            if not mul_std==0:
                corr.append(np.cov(x,y)[0][1]/mul_std)
    avg_corr=sum(corr)/len(corr)
    del corr
    gc.collect()
    print("avg_corr:%.4f"%avg_corr)
    return avg_corr
device='cuda'
if __name__ =="__main__":
    model=CATMODEL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    df_train=pd.read_pickle('df_train.pkl')
    train_loader = DataLoader(df_train)
    del df_train
    gc.collect()
    df_test=pd.read_pickle('df_test.pkl')
    test_loader = DataLoader(df_test)
    del df_test
    gc.collect()

    for epoch in range(300):
        train_epoch(model, device, train_loader, optimizer, epoch,batch_size=2500)
        corr = test_epoch(model, device, test_loader,batch_size=512)
        if float(corr)>0.1:
            torch.save(model.state_dict(),"namodel4-l2-corr-%.4f.data"%corr)
        