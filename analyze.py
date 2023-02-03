


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


params=[]


for i in range(120):
    print(i)
    model_filepath, scratch_dirpath, examples_dirpath, scale_parameters_filepath=helper.get_paths(i);
    interface=helper.engine(model_filepath,examples_dirpath,scale_parameters_filepath);
    s=list(interface.model.parameters())
    params.append((s[-2].data.cpu(),s[-1].data.cpu()))



