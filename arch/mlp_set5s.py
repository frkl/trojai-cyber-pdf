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
        #nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        #nlayers3=params.nlayers3
        
        #self.encoder1=MLP(2400,nh,nh,nlayers);
        self.encoder2=MLP(4800,nh2,2,nlayers);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[];
        fvs=[fv.to(self.w.device) for fv in data_batch['fvs']];
        fvs=[torch.cat((torch.log(fv.abs()+1e-8)/10,torch.sign(fv)),dim=-1) for fv in fvs];
        #trim 4 layers
        fvs_i=[fv[-4:,:].view(4800) for fv in fvs];
        h=torch.stack(fvs_i,dim=0);
        h=self.encoder2(h);
        
        #h=torch.tanh(h)*8;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
