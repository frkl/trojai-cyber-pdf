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
            h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h);
        return h


def quantile(x,q,dim):
    v=torch.quantile(x,torch.Tensor(q).to(x.device),dim=dim,keepdim=True);
    if dim>=0:
        dim2=dim+1;
    else:
        dim2=dim;
    
    v=v.transpose(0,dim2).contiguous().squeeze(dim=0);
    return v;


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
        
        self.q=torch.Tensor([0,0.25,0.5,0.75,1]);
        self.q=torch.arange(0,2+1).float()/2;
        nh=params.nh;
        #nh2=params.nh2;
        #nh3=params.nh3;
        nlayers=params.nlayers
        #nlayers2=params.nlayers2
        #nlayers3=params.nlayers3
        #self.margin=params.margin
        
        self.encoder=MLP(len(self.q)+2,nh,2,nlayers);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[];
        fvs=torch.stack(data_batch['fvs'],dim=0).to(self.w.device);
        fvs_a=quantile(fvs[:,:-2],self.q,dim=1);
        fvs_b=fvs[:,-2:]
        fvs=torch.cat((fvs_a,fvs_b),dim=-1);
        #print(fvs.min(),fvs.max())
        
        h=self.encoder(fvs);
        
        #h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
