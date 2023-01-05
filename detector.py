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
    
    #Extract dataset from a folder of training models
    def extract_dataset(self,models_dirpath,params):
        data=db.Table({'model_id':[],'model_name':[],'fvs':[],'label':[]}); # label currently incorrect
        data=db.DB({'table_ann':data});
        
        t0=time.time()
        
        default_params=smartparse.obj();
        default_params.nbins=100;
        default_params.szcap=4096;
        default_params.fname='data_cyber-pdf_weight.pt'
        params=smartparse.merge(params,default_params)
        
        data.d['params']=db.Table.from_rows([vars(params)]);
        
        models=os.listdir(models_dirpath);
        models=sorted(models)
        import weight_analysis as feature_extractor
        for i,fname in enumerate(models):
            print(i,fname)
            _,fv=feature_extractor.extract_fv(fname,params=params,scale_parameters_filepath=self.scale_parameters_filepath,root=models_dirpath);
            
            #Load GT
            fname_gt=os.path.join(models_dirpath,fname,'ground_truth.csv');
            f=open(fname_gt,'r');
            for line in f:
                line.rstrip('\n').rstrip('\r')
                label=int(line);
                break;
            
            f.close();
            
            
            data['table_ann']['model_name'].append(fname);
            data['table_ann']['model_id'].append(i);
            data['table_ann']['label'].append(label);
            data['table_ann']['fvs'].append(fv);
            
            print('Model %d(%s), time %f'%(i,fname,time.time()-t0));
        
        return data;
    
    def train(self,crossval_splits,params_):
        max_batch=16;
        arch_=importlib.import_module(params_.arch);
        #Run splits
        t0=time.time();
        nets=[];
        for split_id,split in enumerate(crossval_splits):
            data_train,data_val,data_test=split;
            net=arch_.new(params_).cuda();
            opt=optim.Adam(net.parameters(),lr=params_.lr); #params_.lr
            
            #Training
            for iter in range(params_.epochs):
                #print('iter %d/%d'%(iter,params_.epochs))
                net.train();
                for data_batch in data_train.batches(params_.batch,shuffle=True,full=True):
                    opt.zero_grad();
                    net.zero_grad();
                    data_batch.cuda();
                    C=data_batch['label'];
                    data_batch.delete_column('label');
                    scores_i=net(data_batch);
                    
                    #loss=F.binary_cross_entropy_with_logits(scores_i,C.float());
                    spos=scores_i.gather(1,C.view(-1,1)).mean();
                    sneg=torch.exp(scores_i).mean();
                    loss=-(spos-sneg+1);
                    l2=0;
                    for p in net.parameters():
                        l2=l2+(p**2).sum();
                    
                    loss=loss+l2*params_.decay;
                    loss.backward();
                    opt.step();
            
            net.eval();
            nets.append(net);
        
        #Calibration
        scores=[];
        gt=[];
        for split_id,split in enumerate(crossval_splits):
            data_train,data_val,data_test=split;
            net=nets[split_id];
            for data_batch in data_val.batches(max_batch):
                data_batch.cuda();
                
                C=data_batch['label'];
                data_batch.delete_column('label');
                scores_i=net.logp(data_batch);
                scores.append(scores_i.data);
                gt.append(C);
        
        scores=torch.cat(scores,dim=0);
        gt=torch.cat(gt,dim=0);
        
        T=torch.Tensor(1).fill_(0).cuda();
        T.requires_grad_();
        opt2=optim.Adamax([T],lr=3e-2);
        for iter in range(500):
            opt2.zero_grad();
            loss=F.binary_cross_entropy_with_logits(scores*torch.exp(-T),gt.float().cuda());
            loss.backward();
            opt2.step();
        
        #Eval & store
        scores=[];
        scores_pre=[];
        gt=[]
        model_name=[]
        ensemble=[];
        for split_id,split in enumerate(crossval_splits):
            data_train,data_val,data_test=split;
            net=nets[split_id];
            ensemble.append({'net':net.state_dict(),'params':params_,'T':float(T.data.cpu())})
            for data_batch in data_test.batches(max_batch):
                data_batch.cuda();
                
                C=data_batch['label'];
                data_batch.delete_column('label');
                scores_i=net.logp(data_batch);
                
                scores.append((scores_i*torch.exp(-T)).data.cpu());
                scores_pre.append(scores_i.data.cpu());
                model_name=model_name+data_batch['model_name'];
                
                gt.append(C.data.cpu());
        
        scores=torch.cat(scores,dim=0);
        scores_pre=torch.cat(scores_pre,dim=0);
        gt=torch.cat(gt,dim=0);
        
        def compute_metrics(scores,gt,keys=None):
            #Overall
            auc=float(sklearn.metrics.roc_auc_score(torch.LongTensor(gt).numpy(),torch.Tensor(scores).numpy()));
            #ce_=float(F.binary_cross_entropy_with_logits(torch.Tensor(scores),torch.Tensor(gt)));
            sgt=F.logsigmoid(torch.Tensor(scores)*(torch.Tensor(gt)*2-1))
            ce=-sgt.mean();
            cestd=sgt.std()/len(sgt)**0.5;
            return auc,ce,cestd;
        
        auc,ce,cestd=compute_metrics(scores.tolist(),gt.tolist());
        _,cepre,ceprestd=compute_metrics(scores_pre.tolist(),gt.tolist());
        mistakes=[];
        for i in range(len(gt)):
            if int(gt[i])==1 and float(scores[i])<=0:
                mistakes.append(model_name[i]);
        
        mistakes=sorted(mistakes);
        
        print('AUC: %f, CE: %f + %f, CEpre: %f + %f, time %.2f'%(auc,ce,cestd,cepre,ceprestd,time.time()-t0));
        print('Mistakes: '+','.join(['%s'%i for i in mistakes]));
        print('\n')
        
        return ensemble
        
    
    
    #Retraining logic
    def automatic_configure(self, models_dirpath: str):
        #Extract dataset
        #data=self.extract_dataset(models_dirpath,params=self.params)
        data='data_cyber-pdf_weight.pt'
        #Create dataloader
        import dataloader
        data=dataloader.new(data)
        
        #In case features are a list, send tensors to cuda ahead of time to speed up training
        data.cuda()
        for k in data.data['table_ann'].d.keys():
            if isinstance(data.data['table_ann'][k],list):
                if len(data.data['table_ann'][k])>0 and torch.is_tensor(data.data['table_ann'][k][0]):
                    print('sending to cuda')
                    for i in range(len(data.data['table_ann'][k])):
                        data.data['table_ann'][k][i]=data.data['table_ann'][k][i].cuda();
        
        #Create crossval folds
        folds=data.generate_crossval_folds(nfolds=self.params.nsplits);
        folds+=data.generate_crossval_folds(nfolds=self.params.nsplits);
        folds+=data.generate_crossval_folds(nfolds=self.params.nsplits);
        folds+=data.generate_crossval_folds(nfolds=self.params.nsplits);
        crossval_splits=[(data_train,data_test,data_test) for data_train,data_test in folds] #train val test
        
        #Perform training
        ensemble=self.train(crossval_splits,self.params)
        torch.save(ensemble,os.path.join(self.learned_parameters_dirpath,'model.pt'))
        return True
    
    def manual_configure(self, models_dirpath: str):
        return self.automatic_configure(models_dirpath)
    
    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        
        
        import weight_analysis as feature_extractor
        import util.db as db
        import math
        
        _,fv=feature_extractor.extract_fv(model_filepath=model_filepath, scratch_dirpath=scratch_dirpath, examples_dirpath=examples_dirpath, scale_parameters_filepath=self.scale_parameters_filepath)
        #fvs=db.Table.from_rows([{'fvs':fv}]);
        fvs=db.Table({'fvs':[fv]})
        
        import importlib
        import pandas
        logging.info('Running Trojan classifier')
        if not self.learned_parameters_dirpath is None:
            checkpoint=os.path.join(self.learned_parameters_dirpath,'model.pt')
            try:
                checkpoint=torch.load(os.path.join(self.learned_parameters_dirpath,'model.pt'));
            except:
                checkpoint=torch.load(os.path.join('/',self.learned_parameters_dirpath,'model.pt'));
            
            #Compute ensemble score 
            scores=[];
            for i in range(len(checkpoint)):
                params_=checkpoint[i]['params'];
                arch_=importlib.import_module(params_.arch);
                net=arch_.new(params_);
                
                net.load_state_dict(checkpoint[i]['net'],strict=True);
                #net=net.cuda();
                net.eval();
                
                s_i=net.logp(fvs).data.cpu();
                s_i=s_i#*math.exp(-checkpoint[i]['T']);
                scores.append(float(s_i))
            
            scores=sum(scores)/len(scores);
            scores=torch.sigmoid(torch.Tensor([scores])); #score -> probability
            trojan_probability=float(scores);
        else:
            trojan_probability=0.5;
        
        with open(result_filepath, "w") as fp:
            fp.write('%f'%trojan_probability)

        logging.info("Trojan probability: %f", trojan_probability)
        return trojan_probability
