import os
import torch
import torch.nn.functional as F
import sys


data=torch.load('data_cyber-pdf_weight.pt');



ind=torch.randperm(120).tolist()
ind=torch.arange(0,120).long().tolist()
label=[];
pred=[];
for i in ind:
    label.append(data['table_ann']['label'][i])
    os.system('python entrypoint.py infer \
--model_filepath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/models/id-%08d/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/models/id-%08d/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--scale_parameters_filepath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/scale_params.npy'%(i,i))
    with open('scratch/output.txt','r') as f:
        for line in f:
            p=float(line);
            break;
    
    pred.append(p);
    
    for j in range(len(pred)):
        print(j,pred[j],label[j])
    
    print('Perf',F.binary_cross_entropy(torch.Tensor(pred),torch.Tensor(label)))
    sys.stdout.flush()