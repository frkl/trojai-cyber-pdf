


import os
import sys
import torch
import torch.nn.functional as F
import time
import json
import jsonschema

import math
from sklearn.cluster import AgglomerativeClustering


import util.db as db
import util.smartparse as smartparse
import helper_cyber_pdf as helper

#Quantile pooling
#value at top x percentile, x in 0~100
def hist_v(w,bins=100):
    s,_=w.sort(dim=0);
    wmin=float(w.min());
    wmax=float(w.max());
    
    n=s.shape[0];
    hist=torch.Tensor(bins);
    for i in range(bins):
        x=math.floor((n-1)/(bins-1)*i)
        x=max(min(x,n),0);
        v=float(s[x]);
        hist[i]=v;
    
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



def run(interface,nbins=100,szcap=4096):
    fvs=[];
    data=interface.load_examples();
    examples=data['fvs']
    labels=data['labels']
    for i,param in enumerate(interface.model.parameters()):
        fvs=fvs+analyze(param.data.cuda(),nbins=nbins,szcap=szcap);
    
    fvs_examples=[];
    for i in range(len(examples)):
        fvs_example_i=[]
        interface.model.zero_grad()
        #print(examples[i:i+1])
        print('Example %d:'%i,examples[i:i+1].view(-1).max());
        scores,pred=interface.inference(examples[i:i+1].cuda());
        #print(scores.shape,labels[i:i+1])
        print('Score %d:'%i,scores.view(-1).max());
        logp=F.log_softmax(scores,dim=1);
        loss=F.cross_entropy(logp,torch.LongTensor([labels[i]]).cuda());
        loss.backward();
        
        grads=[param.grad for param in interface.model.parameters()]
        
        for j,param in enumerate(grads):
            print('Grad %d-%d:'%(i,j),param.data.view(-1).max());
            fvs_example_i=fvs_example_i+analyze(param.data.cuda(),nbins=nbins,szcap=szcap);
            print('fv %d-%d:'%(i,j),fvs_example_i[-1].data.view(-1).max());
        
        fvs_examples.append(torch.stack(fvs_example_i,dim=0));
        
    
    fvs=torch.stack(fvs,dim=0);
    fvs_examples=torch.stack(fvs_examples,dim=0);
    print(fvs.shape,fvs_examples.shape)
    return fvs,fvs_examples;

def quantile(x,q,dim):
    v=torch.quantile(x,torch.Tensor(q).to(x.device),dim=dim,keepdim=True);
    if dim>=0:
        dim2=dim+1;
    else:
        dim2=dim;
    
    v=v.transpose(0,dim2).contiguous().squeeze(dim=0);
    return v;



#Fuzzing call for TrojAI R9
def extract_fv(id=None, model_filepath=None, scratch_dirpath=None, examples_dirpath=None, scale_parameters_filepath=None,root=None,params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.nbins=100
    default_params.szcap=4096
    params = smartparse.merge(params,default_params);
    
    if not id is None:
        if not root is None:
            model_filepath, scratch_dirpath, examples_dirpath, scale_parameters_filepath_=helper.get_paths(id,root=root);
        else:
            model_filepath, scratch_dirpath, examples_dirpath, scale_parameters_filepath_=helper.get_paths(id);
        
        if scale_parameters_filepath is None:
            scale_parameters_filepath=scale_parameters_filepath_
    
    interface=helper.engine(model_filepath,examples_dirpath,scale_parameters_filepath);
    p=list(interface.model.parameters());
    
    
    fv=[quantile(x.view(-1),[0,0.25,0.5,0.75,1],dim=0) for x in p];
    fv=torch.cat(fv,dim=0);
    
    #fv=torch.cat((p[-2].data.cpu().view(-1),p[-1].data.cpu().view(-1)),dim=0);
    
    print('Weight analysis done, time %.2f'%(time.time()-t0))
    return fv


if __name__ == "__main__":
    data=db.Table({'model_id':[],'model_name':[],'fvs':[],'label':[]}); # label currently incorrect
    data=db.DB({'table_ann':data});
    
    t0=time.time()
    
    default_params=smartparse.obj();
    default_params.nbins=100;
    default_params.szcap=4096;
    default_params.fname='data_cyber-pdf_weight_v2.2.pt'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    data.d['params']=db.Table.from_rows([vars(params)]);
    
    model_ids=list(range(0,120))
    
    for i,id in enumerate(model_ids):
        print(i,id)
        fv=extract_fv(id,params=params);
        
        
        #Load GT
        fname=os.path.join(helper.get_root(id),'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        
        data['table_ann']['model_name'].append('id-%08d'%id);
        data['table_ann']['model_id'].append(id);
        data['table_ann']['label'].append(label);
        data['table_ann']['fvs'].append(fv);
        
        print('Model %d(%d), time %f'%(i,id,time.time()-t0));
        if i%1==0:
            data.save(params.fname);
    
    data.save(params.fname);

