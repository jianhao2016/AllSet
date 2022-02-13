#!/usr/bin/env python
# coding: utf-8

# In[45]:


import torch
import pickle
import os
import ipdb

import os.path as osp

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from load_other_datasets import * 


def save_data_to_pickle(data, p2root = '../data/', file_name = None):
    '''
    if file name not specified, use time stamp.
    '''
#     now = datetime.now()
#     surfix = now.strftime('%b_%d_%Y-%H:%M')
    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, root = '../data/hypergraph_dataset_updated/', name = None, 
                 p2raw = None,
                 train_percent = 0.01,
                 feature_noise = None,
                 transform=None, pre_transform=None):
        
        existing_dataset = ['20newsW100', 'ModelNet40', 'zoo', 
                            'NTU2012', 'Mushroom', 
                            'coauthor_cora', 'coauthor_dblp',
                            'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                            'walmart-trips-100', 'house-committees-100',
                            'cora', 'citeseer', 'pubmed']
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name
        
        self.feature_noise = feature_noise
        
        self._train_percent = train_percent

        
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        
        if not osp.isdir(root):
            os.makedirs(root)
            
        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name, 'processed')
        
        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)

        
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent.item()
        
    # @property
    # def raw_dir(self):
    #     return osp.join(self.root, self.name, 'raw')

    # @property
    # def processed_dir(self):
    #     return osp.join(self.root, self.name, 'processed')


    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features


    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                print(p2f)
                print(self.p2raw)
                print(self.name)

                if self.name in ['cora', 'citeseer', 'pubmed']:
                    tmp_data = load_citation_dataset(path = self.p2raw,
                            dataset = self.name, 
                            train_percent = self._train_percent)

                elif self.name in ['coauthor_cora', 'coauthor_dblp']:
                    assert 'coauthorship' in self.p2raw
                    dataset_name = self.name.split('_')[-1]
                    tmp_data = load_citation_dataset(path = self.p2raw,
                            dataset = dataset_name,
                            train_percent = self._train_percent)

                elif self.name in ['amazon-reviews', 'walmart-trips', 'house-committees']:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')
                    tmp_data = load_cornell_dataset(path = self.p2raw,
                        dataset = self.name,
                        feature_noise = self.feature_noise,
                        train_percent = self._train_percent)
                elif self.name in ['walmart-trips-100', 'house-committees-100']:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')
                    feature_dim = int(self.name.split('-')[-1])
                    tmp_name = '-'.join(self.name.split('-')[:-1])
                    tmp_data = load_cornell_dataset(path = self.p2raw,
                        dataset = tmp_name,
                        feature_dim = feature_dim,
                        feature_noise = self.feature_noise,
                        train_percent = self._train_percent)


                elif self.name == 'yelp':
                    tmp_data = load_yelp_dataset(path = self.p2raw,
                            dataset = self.name,
                            train_percent = self._train_percent)

                else:
                    tmp_data = load_LE_dataset(path = self.p2raw, 
                                              dataset = self.name,
                                              train_percent= self._train_percent)
                    
                _ = save_data_to_pickle(tmp_data, 
                                          p2root = self.myraw_dir,
                                          file_name = self.raw_file_names[0])
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':

    p2root = '../data/hypergraph_dataset_updated/'
    p2raw = '../data/raw_data/'
    # dd = dataset_Hypergraph(root = p2root, name = 'walmart-trips-100', feature_noise = 0, 
    #         p2raw = p2raw)

    for f in ['walmart-trips-100', ]:# 'house-committees', 'amazon-reviews']:
        for feature_noise in [0.1, 1]:
            dd = dataset_Hypergraph(root = p2root, 
                    name = f,
                    feature_noise = feature_noise,
                    p2raw = p2raw)

            assert dd.data.num_nodes in dd.data.edge_index[0]
            print(dd, dd.data)
