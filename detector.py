import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

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

import os
import time
import torch
import math
import sklearn
import importlib
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import util.smartparse as smartparse
import util.db as db
import copy

import trinity as trinity
import crossval
import helper_cyber_pdf as helper

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        self.params=smartparse.dict2obj(metaparameters)
        
        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)
    
    
    #Retraining logic
    def automatic_configure(self, models_dirpath: str):
        dataset=trinity.extract_dataset(models_dirpath,ts_engine=trinity.ts_engine,params=self.params)
        splits=crossval.split(dataset,self.params)
        ensemble=crossval.train(splits,self.params)
        torch.save(ensemble,os.path.join(self.learned_parameters_dirpath,'model.pt'))
        return True
    
    def manual_configure(self, models_dirpath: str):
        return self.automatic_configure(models_dirpath)
    
    def infer(self,model_filepath,result_filepath,scratch_dirpath,examples_dirpath,round_training_dataset_dirpath):
        #Instantiate interface
        params=smartparse.obj()
        params.model_filepath=model_filepath;
        params.examples_dirpath=examples_dirpath;
        params.scale_parameters_filepath=self.scale_parameters_filepath;
        interface=helper.engine(params=params)
        
        #Extract features
        fvs=trinity.extract_fv(interface,trinity.ts_engine,self.params);
        fvs=db.Table.from_rows([fvs]);
        
        #Load model
        if not self.learned_parameters_dirpath is None:
            try:
                ensemble=torch.load(os.path.join(self.learned_parameters_dirpath,'model.pt'));
            except:
                ensemble=torch.load(os.path.join('/',self.learned_parameters_dirpath,'model.pt'));
            
            #print(ensemble)
            trojan_probability=trinity.predict(ensemble,fvs)
            
        else:
            trojan_probability=0.5;
        
        print(trojan_probability)
        
        with open(result_filepath, "w") as fp:
            fp.write('%f'%trojan_probability)
        
        logging.info("Trojan probability: %f", trojan_probability)
        return trojan_probability
