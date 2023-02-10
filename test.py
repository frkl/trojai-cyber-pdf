import os
import torch

import util.db as db
import helper_cyber_pdf as helper

#

models=os.listdir(os.path.join(helper.root(),'models'))
models=sorted(models)

all_data=[];
for m in models[:60]:
    interface=helper.engine(os.path.join(helper.root(),'models',m))
    data=interface.load_examples()
    data=list(data.rows())
    
    try:
        data2=interface.load_examples(os.path.join(helper.root(),'models',m,'poisoned-example-data'))
        data2=list(data2.rows())
        data+=data2
    except:
        pass;
    
    all_data+=data;

all_data=db.Table.from_rows(all_data)


#Try to remove data entries that are dupicated


torch.save(all_data,'enum.pt')

ind=[0]
import torch
import torch.nn.functional as F
for i in range(1,len(all_data)):
    v=all_data['fvs'][i]
    d=((all_data['fvs'][ind]-v)**2).sum(-1);
    if d.min()<1e-8:
        pass;
    else:
        ind.append(i)


torch.save(list(all_data.select_by_index(ind).rows()),'enum.pt')