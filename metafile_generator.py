import torch

#Load current hyperparameters
checkpoint='learned_parameters/model.pt'
tmp=torch.load(checkpoint)
params=vars(tmp[0]['params'])
params.pop('session')
params.pop('session_dir')
params.pop('argv')
params.pop('tag')
params.pop('budget')
params.pop('stuff')

#Common hyperparam template
hp_config={};
hp_config['arch']=['str',[params['arch']]]
hp_config['nh']=['int',16,512]
hp_config['nh2']=['int',16,512]
hp_config['nh3']=['int',16,512]

hp_config['nlayers']=['int',1,12]
hp_config['nlayers2']=['int',1,12]
hp_config['nlayers3']=['int',1,12]

hp_config['margin']=['float',2,10]
hp_config['epochs']=['int',3,500]
hp_config['lr']=['float',1e-5,1e-2]
hp_config['decay']=['float',1e-8,1e-3]
hp_config['batch']=['int',8,64]

hp_config['nsplits']=['int',1,10]
hp_config['pct']=['float',0.5,1.0]

#Generate metaparam template
template={};
for k in hp_config:
    template_k={'description':k}
    if hp_config[k][0]=='str':
        template_k['type']='string'
        template_k['enum']=hp_config[k][1]
    elif hp_config[k][0]=='int':
        template_k['type']='integer'
        template_k['minimum']=hp_config[k][1]
        template_k['maximum']=hp_config[k][2]
        template_k['suggested_minimum']=hp_config[k][1]
        template_k['suggested_maximum']=hp_config[k][2]
    elif hp_config[k][0]=='float':
        template_k['type']='number'
        template_k['suggested_minimum']=hp_config[k][1]
        template_k['suggested_maximum']=hp_config[k][2]
    
    template[k]=template_k

params2={};
for k in hp_config:
    params2[k]=params[k]

import json
with open('metaparameters_schema_.json','r') as f:
    current_template=json.load(f)

current_template['properties']=template

with open('metaparameters_schema.json','w') as f:
    json.dump(current_template,f)


with open('metaparameters.json','w') as f:
    json.dump(params2,f)


