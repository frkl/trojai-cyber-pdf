import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        
        self.layers=nn.ModuleList();
        self.bn=nn.LayerNorm(ninput);
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h).view(*(list(x.shape[:-1])+[-1]));
        return h





class encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.layers=nn.ModuleList();
        for i in range(nlayers):
            if i==0:
                self.layers.append(nn.Conv2d(ninput,nh,5,padding=2));
            else:
                self.layers.append(nn.Conv2d(nh,nh,5,padding=2));
        
    
    
    def forward(self,x):
        h=x;
        for i,layer in enumerate(self.layers):
            if i>0:
                h=F.relu(h);
                h=h+layer(h)
            else:
                h=layer(h)
        
        h=h.mean(dim=-1).mean(dim=-1);
        return h

class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        
        
        nh=params.nh;
        nh2=params.nh2;
        nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        nlayers3=params.nlayers3
        self.margin=params.margin
        
        net=torchvision.models.resnet18()
        self.encoder=nn.Sequential(*list(net.children())[:-1])
        
        #self.encoder1=MLP(512,nh,nh2,nlayers);
        #self.encoder1=MLP(2400,512,512,4);
        #self.encoder2=MLP(nh,nh2,nh2,nlayers2);
        #self.encoder3=MLP(512,nh3,2,nlayers2);
        self.encoder3=nn.Linear(512,2);
        #self.encoder3=MLP(512,512,2,4);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[];
        #fvs=[fv.to(self.w.device) for fv in data_batch['fvs']];
        fvs=data_batch['fvs']
        #print(fvs[0].shape)
        fvs=[fv[-3:,:,:].cuda().contiguous() for fv in fvs];
        fvs=torch.stack(fvs,dim=0);
        ind=list(range(350))+(torch.randperm(20)+350).tolist()
        fvs=fvs[:,:,ind]
        fvs=fvs[:,:,:,ind]
        h=self.encoder(fvs).view(-1,512)
        #print(h.shape)
        #fvs=torch.log1p(fvs.abs()*1e5)*torch.sign(fvs);
        #h=self.encoder1(fvs).mean(dim=-2);
        h=self.encoder3(h);
        
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
