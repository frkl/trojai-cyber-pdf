


import os
import sys
import torch
import torch.nn.functional as F
import torch.autograd.functional as AF
import time
import json
import jsonschema
from torch.autograd import grad

import math
from sklearn.cluster import AgglomerativeClustering


import util.db as db
import util.smartparse as smartparse
import helper_cyber_pdf as helper

#Quantile pooling
#value at top x percentile, x in 0~100
def hist_v(w,bins=100):
    q=torch.arange(bins).float().cuda()/(bins-1)
    hist=torch.quantile(w.view(-1).float(),q.to(w.device)).contiguous().cpu();
    return hist;

def analyze(w,bins=100):
    v0=hist_v(w.view(-1),bins=100)
    v1=hist_v(w.abs().view(-1),bins=100)
    v=torch.cat([v0,v1],dim=0).data.cpu();
    return v;


def run(data,classes,interface,params):
    #Then compute gradients on datapoints
    fvs=[];
    for i in range(len(data)):
        for c in classes:
            print('Grad example %d-%d   '%(i,c), end='\r');
            interface.model.zero_grad();
            x=data[i]['fvs'].clone().cuda()
            x.requires_grad_();
            pred=interface.model(x)
            #pred=interface.inference(x);
            
            logp=F.log_softmax(pred.unsqueeze(0),dim=1);
            loss=F.cross_entropy(logp,torch.LongTensor([c]).cuda());
            
            dx=grad(loss,[x],create_graph=True)
            loss=(dx[0]**2).sum()
            
            #loss=pred[c]
            loss.backward()
            
            fv=[];
            with torch.no_grad():
                for param in interface.model.parameters(): 
                    fv.append(analyze(param.grad.data,bins=params.nbins))
            
            fv=torch.stack(fv,dim=0);
            fvs.append({'fvs':fv,'target':c,'example_id':i});
    
    fvs=db.Table.from_rows(fvs).d
    return fvs;
    

#Fuzzing call for TrojAI R9
def extract_fv(id=None, model_filepath=None, scratch_dirpath=None, examples_dirpath=None, scale_parameters_filepath=None,root=None,params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.nbins=100
    default_params.szcap=4096
    params = smartparse.merge(params,default_params);
    
    interface=helper.engine(id,root=root);
    data=interface.load_examples()
    
    #First inference on data
    classes=[];
    for i in range(len(data)):
        pred=interface.inference(data[i]);
        c=pred.max(dim=0)[1];
        classes.append(int(c));
    
    classes=sorted(list(set(classes)));
    print(classes)
    
    #Run 
    
    fvs=run(data,classes,interface,params)
    print('Grad analysis done, time %.2f'%(time.time()-t0))
    return fvs


if __name__ == "__main__":
    data=db.Table({'model_id':[],'model_name':[],'fvs':[],'label':[]}); # label currently incorrect
    data=db.DB({'table_ann':data});
    
    t0=time.time()
    
    default_params=smartparse.obj();
    default_params.nbins=100;
    default_params.world_size=1;
    default_params.rank=0;
    default_params.fname='data_cyber-pdf_gradv2'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    data.d['params']=db.Table.from_rows([vars(params)]);
    
    model_ids=list(range(0,120))
    model_ids=model_ids[params.rank::params.world_size];
    try:
        os.mkdir(params.fname)
    except:
        pass
    
    for i,id in enumerate(model_ids):
        print('rank %d/%d, model %d/%d, %d'%(params.rank,params.world_size,i,len(model_ids),id))
        fv=extract_fv(id,params=params);
        
        #Load GT
        fname=os.path.join(helper.get_root(id),'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        
        data={'model_name':'id-%08d'%id,'model_id':id,'label':label,'param':vars(params)}
        data.update(fv);
        torch.save(data,'%s/%d.pt'%(params.fname,id));
        
        print('Model %d(%d), time %f'%(i,id,time.time()-t0));
    

