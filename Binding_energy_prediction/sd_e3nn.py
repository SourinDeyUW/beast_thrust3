#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.chdir('phononDoS_tutorial/')

print(os.getcwd)
# In[19]:




# In[2]:


# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
import e3nn,pandas as pd
from e3nn import o3
from typing import Dict, Union

# crystal structure data
from ase import Atom, Atoms,io
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
from pymatgen.core.periodic_table import Element,Specie
# data pre-processing and visualization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import HTML
from pymatgen.core.structure import Structure
# utilities
import time,glob,os,re
from tqdm import tqdm
from utils.utils_data import (load_data, train_valid_test_split, plot_example, plot_predictions, plot_partials,
                              palette, colors, cmap)
from utils.utils_model import Network, visualize_layers, train
from utils.utils_plot import plotly_surface, plot_orbitals, get_middle_feats

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

name='surface_v1_'
ep=200
# In[43]:


# cif2=[]
# phdos=[]
# df=pd.read_csv('cifs//adsorbed_summary.csv')
# os.chdir('cifs')
# for file in glob.glob("*cif"):
# #     print(file)
#     s=file
#     s,_=file.split('.cif')
#     s=re.sub('[!.@#$+-]/', '', str(s))
#     locs=np.where(df['material_id']==int(s))
# #     print(s,locs[0])
#     phdos.append(df['energy'][locs[0]].values)
#     with open(file) as ifile:
#         lines=ifile.readlines()
#         lines=[x.strip() for x in lines]
#     cif2.append(lines) 
# phdos=torch.tensor(phdos)
# os.chdir('..')








#d1=pd.read_csv('adsorbed_summary.csv')
#d1


# In[6]:


from ase import io
os.chdir('surfaces')

d1=pd.read_csv('surface_summary.csv')

f=glob.glob('*cif*')
type_encoding = {}
r_max=5
specie_am = []
for Z in tqdm(range(1, 119), bar_format=bar_format):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)
#     print(specie,specie.mass)
type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))
db=[]
counter=0
import sys
for c in  f:
    cif=io.read(c)
    symbols=list(cif.symbols).copy()
    positions = torch.tensor(cif.positions.copy())
    lattice = torch.tensor(cif.cell.array.copy()).unsqueeze(0)
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=cif, cutoff=r_max, self_interaction=True)
#     print(cif)
    
#     sys.exit(1)
    
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.tensor(edge_src)]
    edge_vec = (positions[torch.tensor(edge_dst)]
                - positions[torch.tensor(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    s,_=c.split('.')
    ix=np.where(d1['material_id']==int(s))
    pho=d1['binding_energy'][ix[0][0]]
    data=tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        phdos=torch.tensor(pho).unsqueeze(0)
    )
    db.append(data)
    counter+=1
data=db
os.chdir('..')


# In[7]:


len(data)


# In[8]:


arr=np.arange(149)
np.random.shuffle(arr)
index_train=list(arr[:119])
index_test=list(arr[119:134])
index_valid=list(arr[134:])
assert set(index_train).isdisjoint(set(index_test))
assert set(index_train).isdisjoint(set(index_valid))
assert set(index_test).isdisjoint(set(index_valid))
batch_size = 1
dataloader_train = tg.loader.DataLoader([db[k] for k in index_train], batch_size=batch_size, shuffle=True)
dataloader_valid = tg.loader.DataLoader([db[k] for k in index_valid], batch_size=batch_size)
dataloader_test = tg.loader.DataLoader([db[k] for k in index_test], batch_size=batch_size)


# In[9]:


# calculate average number of neighbors
def get_neighbors(df, idx):
    n = []
    for entry in db:
#         print(entry.pos)
        N = entry.pos.shape[0]
        for i in range(N):
            n.append(len((entry.edge_index[0] == i).nonzero()))
    return np.array(n)
# idx_train, idx_valid, idx_test = train_valid_test_split(df, species, valid_size=.1, test_size=.1, seed=12, plot=False)

n_train = get_neighbors(db, index_train)


# In[10]:


class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
#         print(output)
#         output = torch.relu(output)
        
        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)  # take mean over atoms per example
        
#         maxima, _ = torch.max(output, dim=1)
#         output = output.div(maxima.unsqueeze(1))
        
        return output


# In[11]:


out_dim = 1
em_dim = 64  

model = PeriodicNetwork(
    in_dim=118,                            # dimension of one-hot encoding of atom type
    em_dim=em_dim,                         # dimension of atom-type embedding
    irreps_in=str(em_dim)+"x0e",           # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    irreps_out=str(out_dim)+"x0e",         # out_dim scalars (L=0 and even parity) to output
    irreps_node_attr=str(em_dim)+"x0e",    # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    layers=2,                              # number of nonlinearities (number of convolutions = layers + 1)
    mul=32,                                # multiplicity of irreducible representations
    lmax=1,                                # maximum order of spherical harmonics
    max_radius=r_max,                      # cutoff radius for convolution
    num_neighbors=n_train.mean(),          # scaling factor based on the typical number of neighbors
    reduce_output=True                     # whether or not to aggregate features of all atoms at the end
)

# print(model)


# In[12]:


torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print('torch device:' , device)


# In[14]:


opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)
run_name=name
#run_name = 'model_' + time.strftime("%y%m%d", time.localtime())
print(run_name)
model.pool = True


# In[15]:


train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name,
      max_iter=ep, scheduler=scheduler, device=device)


# In[18]:


model.load_state_dict(torch.load(run_name + '.torch', map_location=device)['state'])
model.pool = True

df=pd.DataFrame(columns=['y_test','y_pred'])
dataloader = dataloader_test
#df['mse'] = 0.
#df['phdos_pred'] = np.empty((len(df), 0)).tolist()

model.to(device)
model.eval()
with torch.no_grad():
    i0 = 0
    for i, d in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
        d.to(device)
        output = model(d)
        print("hoa")
        loss = F.mse_loss(output, d.phdos, reduction='none').mean(dim=-1).cpu().numpy()
        #print(output,d.phdos)
        #df.loc[i0,:]=d.phdos,output
        df.loc[i0,:]=d.phdos.cpu().numpy()[0],output.cpu().numpy()[0][0]
        #print(output.detach().numpy()[0][0],d.phdos.detach().numpy()[0])
        i0+=1

# In[16]:
name=name+str(ep)+'_epochs.csv'

df.to_csv(name,index=False)

