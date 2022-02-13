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

from convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_scatter import scatter_add, scatter

dname_list = ['cora', 'citeseer', 'pubmed',
              'coauthor_cora', 'coauthor_dblp',
              'NTU2012', 'ModelNet40', 
              'zoo', 'Mushroom', '20newsW100', 
              'yelp', 'house-committees-100', 'walmart-trips-100']

idx_list = ['num_node', 'num_he', 'num_feature', 'num_class', 
              'max_he_size', 'min_he_size', 'avg_he_size', 'median_he_size',
              'max_node_degree', 'min_node_degree', 'avg_node_degree', 'median_node_degree']

stats_df = pd.DataFrame(columns = dname_list, index = idx_list)
feature_noise = 1

def get_stats(deg_list):
    tmp_list = deg_list.numpy()
    return [np.max(tmp_list), np.min(tmp_list), np.mean(tmp_list), np.median(tmp_list)]

for dname in tqdm(dname_list):
    if dname not in ['house-committees-100', 'walmart-trips-100']:
        ds = dataset_Hypergraph(name = dname)
    else:
        ds = dataset_Hypergraph(name = dname, feature_noise = feature_noise)
    
    data = ds.data
    
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    num_classes = len(data.y.unique())
    
    c_idx = th.where(data.edge_index[0] == num_nodes)[0].min()
    V2E = data.edge_index[:, :c_idx]
    
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

# stats_df.to_csv('datasets_stats.csv')
print(stats_df)
    
