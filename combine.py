import torch
import os
import util.db as db

root='data_cyber-pdf_gradv2'

fvs=[torch.load(os.path.join(root,'%d.pt'%id)) for id in range(120)]
table_ann=db.Table.from_rows(fvs)
print(table_ann[0]['fvs'].shape)
#table_ann.delete_column('fvs_color2')
#table_ann.delete_column('fvs2')
data=db.DB({'table_ann':table_ann});

data.save('%s.pt'%root)
