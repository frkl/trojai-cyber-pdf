#Python2,3 compatible headers
from __future__ import unicode_literals,division
from builtins import int
from builtins import range

#System packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import math
import time
import random
import argparse
import sys
import os
import re
import copy
import importlib
from collections import namedtuple
from collections import OrderedDict
from itertools import chain

import util.db as db
import util.smartparse as smartparse
import util.file
import util.session_manager as session_manager
import dataloader

import sklearn.metrics
from hyperopt import hp, tpe, fmin

# Training settings
def default_params():
    params=smartparse.obj();
    #Data
    params.nsplits=4;
    params.pct=0.5
    #Model
    params.arch='arch.mlp_eig';
    params.data='data_cyber-pdf_weight_v2.2.pt';
    params.tag='';
    #MISC
    params.session_dir=None;
    params.budget=10000;
    return params

def create_session(params):
    session=session_manager.Session(session_dir=params.session_dir); #Create session
    torch.save({'params':params},session.file('params.pt'));
    pmvs=vars(params);
    pmvs=dict([(k,pmvs[k]) for k in pmvs if not(k=='stuff')]);
    print(pmvs);
    util.file.write_json(session.file('params.json'),pmvs); #Write a human-readable parameter json
    session.file('model','dummy');
    return session;


params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;

data=dataloader.new(params.data);
data.cuda();
params.stuff=data.preprocess();

import pandas
meta=pandas.read_csv('../trojai-datasets/cyber-pdf-dec2022-train/METADATA.csv');
meta_table={};
meta_table['model_name']=list(meta['model_name']);
meta_table['label']=[int(x) for x in list(meta['poisoned'])];
#meta_table['model_architecture']=list(meta['model_architecture']);
#meta_table['task_type']=list(meta['task_type']);

#meta_table['trigger_option']=list(meta['trigger_option']);
#meta_table['trigger_type']=list(meta['trigger_type']);
meta_table=db.Table(meta_table)
assert 'model_name' in data.data['table_ann'].d.keys()


data.data['table_ann']=db.left_join(data.data['table_ann'],meta_table,'model_name');

for k in data.data['table_ann'].d.keys():
    if isinstance(data.data['table_ann'][k],list):
        if len(data.data['table_ann'][k])>0 and torch.is_tensor(data.data['table_ann'][k][0]):
            print('sending to cuda')
            for i in range(len(data.data['table_ann'][k])):
                data.data['table_ann'][k][i]=data.data['table_ann'][k][i].cuda();


X=[]
y=[]
for data_batch in data.batches(16):
    data_batch.cuda();
    
    C=data_batch['label'];
    
    for i in range(len(data_batch['fvs'])):
        hi=torch.Tensor(14*5).fill_(0);
        hi[-len(data_batch['fvs'][i]):]=data_batch['fvs'][i]
        X.append(hi)
    
    y.append(C)

X=torch.stack(X,dim=0)
y=torch.cat(y,dim=0);


from sklearn import tree
clf = tree.DecisionTreeClassifier()
sklearn.model_selection.cross_validate(clf,X,y)


