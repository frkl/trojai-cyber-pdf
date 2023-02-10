
import os
import numpy as np
import torch
import pickle
import json
import jsonschema
import warnings

from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s

import torch.nn.functional as F
import util.smartparse as smartparse
import util.db as db


warnings.filterwarnings("ignore")
'''
def get_paths(id,root='../trojai-datasets/cyber-pdf-dec2022-train'):
    if not isinstance(id,str):
        id='models/id-%08d'%id;
    
    if root is None:
        root='../trojai-datasets/cyber-pdf-dec2022-train'
    
    model_filepath=os.path.join(root,id,'model.pt');
    examples_dirpath=os.path.join(root,id,'clean-example-data');
    scratch_dirpath='./scratch'
    scale_parameters_filepath=os.path.join(root,'scale_params.npy');
    
    return model_filepath, scratch_dirpath, examples_dirpath, scale_parameters_filepath;
'''

def root():
    return '../trojai-datasets/cyber-pdf-dec2022-train'


#The user provided engine that our algorithm interacts with
class engine:
    def __init__(self,folder=None,params=None):
        default_params=smartparse.obj();
        default_params.model_filepath='';
        default_params.examples_dirpath='';
        default_params.scale_parameters_filepath='';
        params=smartparse.merge(params,default_params)
        #print(vars(params))
        if params.model_filepath=='':
            params.model_filepath=os.path.join(folder,'model.pt');
        if params.examples_dirpath=='':
            params.examples_dirpath=os.path.join(folder,'clean-example-data');
        if params.scale_parameters_filepath=='':
            params.scale_parameters_filepath=os.path.join(root(),'scale_params.npy')
        
        
        model=torch.load(params.model_filepath);
        self.model=model.cuda()
        self.model.eval();
        
        self.examples_dirpath=params.examples_dirpath
        self.scale_parameters_filepath=params.scale_parameters_filepath
    
    
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
        
        fnames=[fname for fname in os.listdir(examples_dirpath) if os.path.isfile(os.path.join(examples_dirpath,fname)) and fname.endswith('.npy')]
        fnames=sorted(fnames)
        data=[];
        for fname in fnames:
            feature_vector = np.load(os.path.join(examples_dirpath,fname)).reshape(1, -1)
            feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()
            
            ground_tuth_filepath = os.path.join(examples_dirpath,fname) + ".json"
            
            with open(ground_tuth_filepath, 'r') as ground_truth_file:
                ground_truth =  ground_truth_file.readline()
            
            data.append({'fv':feature_vector,'label':int(ground_truth)});
        
        
        return db.Table.from_rows(data);
    
    def eval(self,data):
        scores=self.model(data['fv'].cuda())
        loss=F.cross_entropy(scores,torch.LongTensor([data['label']]).cuda())
        return loss
    
    
