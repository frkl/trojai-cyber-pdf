
import os
import numpy as np
import torch
import json
import jsonschema
import warnings

import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)

from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch



warnings.filterwarnings("ignore")

def get_paths(id,root='../trojai-datasets/cyber-pdf-dec2022-train'):
    if not isinstance(id,str):
        id='models/id-%08d'%id;
    
    model_filepath=os.path.join(root,id,'model.pt');
    examples_dirpath=os.path.join(root,id,'clean-example-data');
    scratch_dirpath='./scratch'
    scale_parameters_filepath=os.path.join(root,'scale_params.npy');
    
    return model_filepath, scratch_dirpath, examples_dirpath, scale_parameters_filepath;

def get_root(id,root='../trojai-datasets/cyber-pdf-dec2022-train'):
    id='id-%08d'%id;
    return os.path.join(root,'models',id);


#The user provided engine that our algorithm interacts with
class engine:
    def __init__(self,model_filepath,examples_dirpath,scale_parameters_filepath):
        model=torch.load(model_filepath);
        self.model=model.cuda()
        #Load a scalar thingy
        self.examples_dirpath=examples_dirpath
        self.scale_parameters_filepath=scale_parameters_filepath
        
        self.model.eval();
    
    def load_examples(self,examples_dirpath=None,scale_parameters_filepath=None):
        if examples_dirpath is None:
            examples_dirpath=self.examples_dirpath
        
        if scale_parameters_filepath is None:
            scale_parameters_filepath=self.scale_parameters_filepath
        
        scaler = StandardScaler()
        scale_params = np.load(scale_parameters_filepath)
        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]
        print('scaler',scaler.mean_[0],scaler.scale_[0])
        
        fvs=[]
        labels=[];
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()
                
                fvs.append(feature_vector)
                
                ground_tuth_filepath = examples_dir_entry.path + ".json"
                
                with open(ground_tuth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()
                
                labels.append(int(ground_truth));
        
        
        fvs=torch.cat(fvs,dim=0);
        return {'fvs':fvs,'labels':labels};
    
    def inference(self,fvs):
        scores=self.model(fvs)
        _,pred=scores.max(dim=0)
        return scores,pred
    
    #bbox: xywh,relative position to border
    def insert_trigger(self,examples,trigger,bbox,eps=1e-8):
        
        N=len(bbox);
        x,y,w,h=bbox[:,0],bbox[:,1],bbox[:,2],bbox[:,3]
        
        trigger=trigger.clamp(min=0,max=1);
        sx=1/(w+eps);
        sy=1/(h+eps);
        tx=-(2*x-1)*sx-1;
        ty=-(2*y-1)*sy-1;
        
        T=torch.stack((sx,x*0,tx,y*0,sy,ty),dim=-1).view(N,2,3).cuda();
        
        grid=F.affine_grid(T,examples.shape)
        overlay=F.grid_sample(trigger.unsqueeze(0).cuda().repeat(N,1,1,1),grid);
        
        triggered_examples=db.Table(copy.deepcopy(examples.d));
        triggered_examples['im']=[];#triggered_examples['im'].clone()
        for i in range(len(examples)):
            im=examples['im'][i].unsqueeze(0)
            
            alpha=overlay[:,3,:,:];
            color=overlay[:,:3,:,:];
            
            im_out=im.cuda()*(1-alpha)+color*alpha;
            triggered_examples['im'].append(im_out.squeeze(0))
            
        
        return triggered_examples;
    
