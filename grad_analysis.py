import os
import sys
import torch
import torch.nn.functional as F
import torch.autograd.functional as AF
import time
import json
import jsonschema

import math
from sklearn.cluster import AgglomerativeClustering


import util.db as db
import util.smartparse as smartparse


#Quantile pooling
#value at top x percentile, x in 0~100
def hist_v(w,bins=100):
    q=torch.arange(bins).float().cuda()/(bins-1)
    hist=torch.quantile(w.view(-1).float(),q.to(w.device)).contiguous().cpu();
    return hist;

#extract features from a weight matrix
def analyze(param,nbins=100,szcap=4096):
    if len(param.shape)==1:
        fvs=analyze(param.view(1,-1),nbins=nbins,szcap=szcap)
        return fvs
    if len(param.shape)==4:
        #Conv filters
        fvs=[];
        for i in range(param.shape[-2]):
            for j in range(param.shape[-1]):
                fv=analyze(param[:,:,i,j],nbins=nbins,szcap=szcap);
                fvs=fvs+fv;
        
        return fvs
        #elif len(params.shape)==3:
    elif len(param.shape)==2:
        #Create an overarching matrix
        nx=param.shape[1]
        ny=param.shape[0]
        n=max(nx,ny);
        n=min(n,szcap)
        m=min(nx,ny);
        z=torch.zeros(n,n).to(param.device);
        z[:min(ny,n),:min(nx,n)]=param.data[:min(ny,n),:min(nx,n)];
        
        #
        e,_=torch.linalg.eig(z);
        
        #Get histogram of abs, real, imaginary
        e2=(e.real**2+e.imag**2)**0.5;
        e2_hist=hist_v(e2,nbins);
        er_hist=hist_v(e.real,nbins);
        ec_hist=hist_v(e.imag,nbins);
        
        #2) histogram of eig persistence
        cm=AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage='single')
        s=torch.stack((e.real,e.imag),dim=1)
        cm=cm.fit(s.cpu())
        d=torch.from_numpy(cm.distances_);
        eig_persist=hist_v(d,nbins)
        
        
        #Get histogram of weight value and abs
        w=param.data.view(-1);
        w_hist=hist_v(w,nbins);
        wabs_hist=hist_v(w.abs(),nbins);
        
        fv=torch.cat((e2_hist,er_hist,ec_hist,eig_persist,w_hist,wabs_hist),dim=0);
        return [fv];
    else:
        return [];



#Fuzzing call for TrojAI R9
import helper_cyber_pdf as helper
def extract_fv(id=None, model_filepath=None, scratch_dirpath=None, examples_dirpath=None, scale_parameters_filepath=None,root=None,params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.nbins=20
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
    
    #Then compute gradients on datapoints
    fvs=[];
    for i in range(len(data)):
        for c in classes:
            print('Grad example %d-%d   '%(i,c), end='\r');
            interface.model.zero_grad();
            pred=interface.inference(data[i]);
            
            logp=F.log_softmax(pred.unsqueeze(0),dim=1);
            loss=F.cross_entropy(logp,torch.LongTensor([c]).cuda());
            
            #loss=pred[c]
            loss.backward()
            
            fv=[];
            with torch.no_grad():
                for param in interface.model.parameters(): 
                    fv+=analyze(param.grad.data,nbins=params.nbins);
            
            fv=torch.stack(fv,dim=0);
            fvs.append({'fvs':fv,'target':c,'example_id':i});
    
    fvs=db.Table.from_rows(fvs).d
    print('Grad analysis done, time %.2f'%(time.time()-t0))
    return fvs


if __name__ == "__main__":
    data=db.Table({'model_id':[],'model_name':[],'fvs':[],'label':[]}); # label currently incorrect
    data=db.DB({'table_ann':data});
    
    t0=time.time()
    
    default_params=smartparse.obj();
    default_params.nbins=20;
    default_params.world_size=1;
    default_params.rank=0;
    default_params.fname='data_cyber-pdf_grad'
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
    

