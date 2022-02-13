# -*- coding: utf-8 -*-
"""
HAN mini-batch training by RandomWalkSampler.
note: This demo use RandomWalkSampler to sample neighbors, it's hard to get all neighbors when valid or test,
so we sampled twice as many neighbors during val/test than training.
"""
import dgl
import time
import numpy
import argparse
import torch
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dgl.nn.pytorch import GATConv

from dgl.sampling import RandomWalkNeighborSampler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from model_hetero import SemanticAttention
from utils import EarlyStopping, set_random_seed, get_binary_mask
from utils import load_data, load_allset_hg_batch


class HANLayer(torch.nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_metapath : number of metapath based sub-graph
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_metapath, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_metapath = num_metapath

    def forward(self, block_list, h_list):
        semantic_embeddings = []

        for i, block in enumerate(block_list):
            semantic_embeddings.append(self.gat_layers[i](block, h_list[i]).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, num_metapath, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_metapath, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_metapath, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)


class HANSampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        self.sampler_list = []
        for metapath in metapath_list:
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            self.sampler_list.append(RandomWalkNeighborSampler(G=g,
                                                               num_traversals=1,
                                                               termination_prob=0,
                                                               num_random_walks=num_neighbors,
                                                               num_neighbors=num_neighbors,
                                                               metapath=metapath))

    def sample_blocks(self, seeds):
        block_list = []
        for sampler in self.sampler_list:
            frontier = sampler(seeds)
            # add self loop
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(torch.tensor(seeds), torch.tensor(seeds))
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)

        return seeds, block_list


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, metapath_list, num_neighbors, features, labels, val_nid, loss_fcn, batch_size):
    model.eval()

    han_valid_sampler = HANSampler(g, metapath_list, num_neighbors=num_neighbors * 2)
    dataloader = DataLoader(
        dataset=val_nid,
        batch_size=batch_size,
        collate_fn=han_valid_sampler.sample_blocks,
        shuffle=False,
        drop_last=False,
        num_workers=1)
    correct = total = 0
    prediction_list = []
    labels_list = []
    with torch.no_grad():
        for step, (seeds, blocks) in enumerate(dataloader):
            h_list = load_subtensors(blocks, features)
            blocks = [block.to(args['device']) for block in blocks]
            hs = [h.to(args['device']) for h in h_list]

            logits = model(blocks, hs)
            loss = loss_fcn(logits, labels[numpy.asarray(seeds)].to(args['device']))
            # get each predict label
            _, indices = torch.max(logits, dim=1)
            prediction = indices.long().cpu().numpy()
            labels_batch = labels[numpy.asarray(seeds)].cpu().numpy()

            prediction_list.append(prediction)
            labels_list.append(labels_batch)

            correct += (prediction == labels_batch).sum()
            total += prediction.shape[0]

    total_prediction = numpy.concatenate(prediction_list)
    total_labels = numpy.concatenate(labels_list)
    micro_f1 = f1_score(total_labels, total_prediction, average='micro')
    macro_f1 = f1_score(total_labels, total_prediction, average='macro')
    accuracy = correct / total

    return loss, accuracy, micro_f1, macro_f1


def load_subtensors(blocks, features):
    h_list = []
    for block in blocks:
        input_nodes = block.srcdata[dgl.NID]
        h_list.append(features[input_nodes])
    return h_list

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]

    if not balance:
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(labeled_nodes))
        val_lb = int(valid_prop*len(labeled_nodes))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx


def main(args):
    # acm data
    dname_list = ['cora', 'citeseer', 'pubmed',
                  'coauthor_cora', 'coauthor_dblp',
                  'NTU2012', 'ModelNet40', 
                  'zoo', 'Mushroom', '20newsW100', 
                  'yelp', 'house-committees-100', 'walmart-trips-100']

    if args['dataset'] == 'ACMRaw':
        # ipdb.set_trace()
        g, features, labels, n_classes, train_nid, val_nid, test_nid, train_mask, \
        val_mask, test_mask = load_data('ACMRaw')
        metapath_list = [['pa', 'ap'], ['pf', 'fp']]
    elif args['dataset'] in dname_list:
        p2raw = '/data/shared/dgl_data_raw/'
        # ipdb.set_trace()
        g, features, labels, n_classes, _, _, _, _, _, _ = load_allset_hg_batch(remove_self_loop = False, 
                p2raw = p2raw, 
                feature_noise  = args['feature_noise'], 
                dname = args['dataset'])
        metapath_list = [['Vs_E', 'E_Vs'], ['Es_V', 'V_Es']]

    else:
        raise NotImplementedError('Unsupported dataset {}'.format(args['dataset']))

    # features = features.to(args['device'])
    # labels = labels.to(args['device'])
    num_nodes = features.shape[0]

    history_acc, history_micro_f1, history_marco_f1 = [], [], []
    run_time_list = []
    for run in range(args['runs']):
        new_split = rand_train_test_idx(labels, args['train_prop'], args['valid_prop'])
        train_nid, val_nid, test_nid = new_split['train'], new_split['valid'], new_split['test']

        train_mask = get_binary_mask(num_nodes, train_nid)
        val_mask = get_binary_mask(num_nodes, val_nid)
        test_mask = get_binary_mask(num_nodes, test_nid)

        # Is it need to set different neighbors numbers for different meta-path based graph?
        num_neighbors = args['num_neighbors']
        han_sampler = HANSampler(g, metapath_list, num_neighbors)
        # Create PyTorch DataLoader for constructing blocks
        dataloader = DataLoader(
            dataset=train_nid,
            batch_size=args['batch_size'],
            collate_fn=han_sampler.sample_blocks,
            shuffle=True,
            drop_last=False,
            num_workers=4)

        model = HAN(num_metapath=len(metapath_list),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=n_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])

        # total_params = sum(p.numel() for p in model.parameters())
        # print("total_params: {:d}".format(total_params))
        # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print("total trainable params: {:d}".format(total_trainable_params))

        stopper = EarlyStopping(patience=args['patience'])
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                     weight_decay=args['weight_decay'])

        start_time = time.time()
        for epoch in range(args['num_epochs']):
            model.train()
            for step, (seeds, blocks) in enumerate(dataloader):
                h_list = load_subtensors(blocks, features)
                blocks = [block.to(args['device']) for block in blocks]
                hs = [h.to(args['device']) for h in h_list]

                logits = model(blocks, hs)
                loss = loss_fn(logits, labels[numpy.asarray(seeds)].to(args['device']))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print info in each batch
                train_acc, train_micro_f1, train_macro_f1 = score(logits, labels[numpy.asarray(seeds)])
                # print(
                #     "Epoch {:d} | loss: {:.4f} | train_acc: {:.4f} | train_micro_f1: {:.4f} | train_macro_f1: {:.4f}".format(
                #         epoch + 1, loss, train_acc, train_micro_f1, train_macro_f1
                #     ))
            val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, metapath_list, num_neighbors, features,
                                                                     labels, val_nid, loss_fn, args['batch_size'])
            early_stop = stopper.step(val_loss.data.item(), val_acc, model)

            # print('Epoch {:d} | Val loss {:.4f} | Val Accuracy {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            #     epoch + 1, val_loss.item(), val_acc, val_micro_f1, val_macro_f1))

            if early_stop:
                break

        stopper.load_checkpoint(model)
        test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, metapath_list, num_neighbors, features,
                                                                     labels, test_nid, loss_fn, args['batch_size'])
        # print('Test loss {:.4f} | Test Accuracy {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        #     test_loss.item(), test_acc, test_micro_f1, test_macro_f1))

        # print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test Acc {:.4f}'.format(
        #     test_loss.item(), test_micro_f1, test_macro_f1, test_acc))
        history_acc.append(100 * test_acc)
        history_micro_f1.append(100 * test_micro_f1)
        history_marco_f1.append(100 * test_macro_f1)

        end_time = time.time()
        run_time_list.append(end_time - start_time)

    print(f'>> Final test acc: {np.mean(history_acc):.2f}, std: {np.std(history_acc):.2f}; \
            test marco f1: {np.mean(history_marco_f1):.2f}, std: {np.std(history_marco_f1):.2f}')

    print(f'>> Train time per run: {np.mean(run_time_list):.2f}, std: {np.std(run_time_list):.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('mini-batch HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_neighbors', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_heads', type=list, default=[8])
    parser.add_argument('--hidden_units', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='ACMRaw')
    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--runs',type = int, default = 20)
    parser.add_argument('--cuda',type = int, default = 0)
    parser.add_argument('--train_prop', type = float, default = 0.5)
    parser.add_argument('--valid_prop', type = float, default = 0.25)
    parser.add_argument('--feature_noise', type = float, default = 1)
    args = parser.parse_args().__dict__
    # set_random_seed(args['seed'])

    main(args)
