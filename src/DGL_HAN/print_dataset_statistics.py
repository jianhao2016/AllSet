#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import torch as th
import pandas as pd
from tqdm import tqdm
import ipdb
import os
import pickle

from convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_scatter import scatter_add, scatter
from torch.nn.functional import one_hot
from scipy import sparse

dname_list = ['cora', 'citeseer', 'pubmed',
              'coauthor_cora', 'coauthor_dblp',
              'NTU2012', 'ModelNet40', 
              'zoo', 'Mushroom', '20newsW100', 
              'house-committees-100', 'walmart-trips-100', 'yelp']

# dname_list = ['house-committees-100', 'walmart-trips-100', 'yelp']
dname_list = ['house-committees-100', 'walmart-trips-100']

# dname_list = ['yelp',]

idx_list = ['num_node', 'num_he', 'num_feature', 'num_class', 
              'max_he_size', 'min_he_size', 'avg_he_size', 'median_he_size',
              'max_node_degree', 'min_node_degree', 'avg_node_degree', 'median_node_degree']

stats_df = pd.DataFrame(columns = dname_list, index = idx_list)
# feature_noise = 1

def get_stats(deg_list):
    tmp_list = deg_list.numpy()
    return [np.max(tmp_list), np.min(tmp_list), np.mean(tmp_list), np.median(tmp_list)]

p2root = '/data/shared/pyg_data/hypergraph_dataset_updated/'
p2raw = '/data/shared//AllSet_all_raw_data/'
p2dgl_data = '/data/shared/dgl_data_raw/'

if not os.path.isdir(p2dgl_data):
    os.makedirs(p2dgl_data)

for dname in tqdm(dname_list):
    for feature_noise in tqdm([0, 0.2, 0.4, 0.6, 0.8]):
        if dname not in ['house-committees-100', 'walmart-trips-100']:
            ds = dataset_Hypergraph(name = dname, root = p2root)
            data = ds.data
        else:
            ds = dataset_Hypergraph(name = dname, root = p2root, feature_noise = feature_noise, 
                    p2raw = p2raw)
            data = ds.data

    
        num_nodes = data.x.shape[0]
        num_features = data.x.shape[1]
        num_classes = len(data.y.unique())
        
        c_idx = th.where(data.edge_index[0] == num_nodes)[0].min()
        V2E = data.edge_index[:, :c_idx]
        E2V = data.edge_index[:, c_idx:]
        
        num_edges = len(V2E[1].unique())
        if 'num_hyperedges' in data:
            num_he = data.num_hyperedges
            if isinstance(num_he, list):
                num_he = num_he[0]

            if num_he != num_edges:
                ipdb.set_trace()
        
        
        edge_weight = th.ones_like(V2E[0])
        Vdeg = scatter_add(edge_weight, V2E[0], dim=0)
        HEdeg = scatter_add(edge_weight, V2E[1] - num_nodes, dim=0)

        V_list = get_stats(Vdeg)
        E_list = get_stats(HEdeg)

        
        num2str = lambda x: f'{int(x)}' if x == int(x) else f'{x:.2f}'
        stat_list = [num_nodes, num_edges, num_features, num_classes] + E_list + V_list 
        stat_list = [num2str(x) for x in stat_list]

        stats_df[dname] = stat_list

        '''
        dgl_data should contains: 
            label: one hot encoded label of vertices and hyperedges.
            feature: feature matrix of vertices and hyperedges.
            VEV: adjacency matrix of V-V with metapath V-E-V.
            EVE: adj matrix of E-E with metapath E-V-E.
            train_idx: index vector in the shape of (1, #train sample), idx of nodes.
        '''
        dgl_data = {}
        V_label = data.y - data.y.min()
        try:
            V_onehot = one_hot(V_label, num_classes)
        except:
            ipdb.set_trace()
        E_onehot = th.zeros((num_edges, num_classes), dtype = V_onehot.dtype)
        VandE_label = th.cat((V_onehot, E_onehot), dim = 0).numpy()
        dgl_data['label'] = sparse.csr_matrix(VandE_label)

        V_features = data.x
        E_features = th.zeros((num_edges, num_features), dtype = V_features.dtype)
        VandE_feature = th.cat((V_features, E_features), dim = 0).numpy()
        dgl_data['feature'] = sparse.csr_matrix(VandE_feature)

        total_num_VandE = num_nodes + num_edges
        incidence = sparse.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index.numpy()), 
                shape = (total_num_VandE, total_num_VandE))
        two_step_adj = incidence @ incidence
        VEV, EVE = two_step_adj.tolil(), two_step_adj.tolil()
        VEV[num_nodes:, num_nodes:] = 0
        EVE[:num_nodes, :num_nodes] = 0
        dgl_data['VEV'] = sparse.csr_matrix(VEV)
        dgl_data['EVE'] = sparse.csr_matrix(EVE)

        # get VvsE and EvsV incidence matrices for batch sampler.
        V2E_reset_edge_id = V2E
        V2E_reset_edge_id[1, :] -= num_nodes
        E2V_reset_edge_id = E2V
        E2V_reset_edge_id[0, :] -= num_nodes
        VvsE = sparse.csr_matrix((np.ones(V2E_reset_edge_id.shape[1]), V2E_reset_edge_id.numpy()), shape = (num_nodes, num_edges))
        EvsV = sparse.csr_matrix((np.ones(E2V_reset_edge_id.shape[1]), E2V_reset_edge_id.numpy()), shape = (num_edges, num_nodes))

        dgl_data['VvsE'] = VvsE
        dgl_data['EvsV'] = EvsV

        random_idx = np.arange(num_nodes)
        np.random.shuffle(random_idx)
        train_stop, val_stop = int(num_nodes * 0.2), int(num_nodes * 0.4)
        dgl_data['train_idx'] = random_idx[: train_stop]
        dgl_data['val_idx'] = random_idx[train_stop: val_stop]
        dgl_data['test_idx'] = random_idx[val_stop: ]

        p2dgl_raw = os.path.join(p2dgl_data, f'{dname}_noise_{feature_noise}_raw.pickle')
        with open(p2dgl_raw, 'wb') as f:
            pickle.dump(dgl_data, f, protocol = 4)


# stats_df.to_csv('datasets_stats.csv')
print(stats_df)
    
