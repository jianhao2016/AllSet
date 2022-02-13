#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""

"""
 
import torch

import numpy as np

from collections import defaultdict, Counter
from itertools import combinations
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def expand_edge_index(data, edge_th=0):
    '''
    args:
        num_nodes: regular nodes. i.e. x.shape[0]
        num_edges: number of hyperedges. not the star expansion edges.

    this function will expand each n2he relations, [[n_1, n_2, n_3], 
                                                    [e_7, e_7, e_7]]
    to :
        [[n_1,   n_1,   n_2,   n_2,   n_3,   n_3],
         [e_7_2, e_7_3, e_7_1, e_7_3, e_7_1, e_7_2]]

    and each he2n relations:   [[e_7, e_7, e_7],
                                [n_1, n_2, n_3]]
    to :
        [[e_7_1, e_7_2, e_7_3],
         [n_1,   n_2,   n_3]]

    and repeated for every hyperedge.
    '''
    edge_index = data.edge_index
    num_nodes = data.n_x[0].item()
    if hasattr(data, 'totedges'):
        num_edges = data.totedges
    else:
        num_edges = data.num_hyperedges[0]

    expanded_n2he_index = []
#     n2he_with_same_heid = []

#     expanded_he2n_index = []
#     he2n_with_same_heid = []

    # start edge_id from the largest node_id + 1.
    cur_he_id = num_nodes
    # keep an mapping of new_edge_id to original edge_id for edge_size query.
    new_edge_id_2_original_edge_id = {}

    # do the expansion for all annotated he_id in the original edge_index
#     ipdb.set_trace()
    for he_idx in range(num_nodes, num_edges + num_nodes):
        # find all nodes within the same hyperedge.
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

#         Trim a hyperedge if its size>edge_th
        if edge_th > 0:
            if size_of_he > edge_th:
                continue

        if size_of_he == 1:
            # there is only one node in this hyperedge -> self-loop node. add to graph.
            #             n2he_with_same_heid.append(selected_he)

            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)

            # ====
#             new_he2n_same_heid = torch.flip(selected_he, dims = [0])
#             he2n_with_same_heid.append(new_he2n_same_heid)

#             new_he2n = torch.flip(selected_he, dims = [0])
#             new_he2n[0] = cur_he_id
#             expanded_he2n_index.append(new_he2n)

            cur_he_id += 1
            continue

        # -------------------------------
#         # new_n2he_same_heid uses same he id for all nodes.
#         new_n2he_same_heid = selected_he.repeat_interleave(size_of_he - 1, dim = 1)
#         n2he_with_same_heid.append(new_n2he_same_heid)

        # for new_n2he mapping. connect the nodes to all repeated he first.
        # then remove those connection that corresponding to the node itself.
        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        # new_edge_ids start from the he_id from previous iteration (cur_he_id).
        new_edge_ids = torch.LongTensor(
            np.arange(cur_he_id, cur_he_id + size_of_he)).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        # build a mapping between node and it's corresponding edge.
        # e.g. {n_1: e_7_1, n_2: e_7_2}
        tmp_node_id_2_he_id_dict = {}
        for idx in range(size_of_he):
            new_edge_id_2_original_edge_id[cur_he_id] = he_idx
            cur_node_id = selected_he[0][idx].item()
            tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id
            cur_he_id += 1

        # create n2he by deleting the self-product edge.
        new_he_select_mask = torch.BoolTensor([True] * new_n2he.shape[1])
        for col_idx in range(new_n2he.shape[1]):
            tmp_node_id, tmp_edge_id = new_n2he[0, col_idx].item(
            ), new_n2he[1, col_idx].item()
            if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
                new_he_select_mask[col_idx] = False
        new_n2he = new_n2he[:, new_he_select_mask]
        expanded_n2he_index.append(new_n2he)


#         # ---------------------------
#         # create he2n from mapping.
#         new_he2n = np.array([[he_id, node_id] for node_id, he_id in tmp_node_id_2_he_id_dict.items()])
#         new_he2n = torch.from_numpy(new_he2n.T).to(device = edge_index.device)
#         expanded_he2n_index.append(new_he2n)

#         # create he2n with same heid as input edge_index.
#         new_he2n_same_heid = torch.zeros_like(new_he2n, device = edge_index.device)
#         new_he2n_same_heid[1] = new_he2n[1]
#         new_he2n_same_heid[0] = torch.ones_like(new_he2n[0]) * he_idx
#         he2n_with_same_heid.append(new_he2n_same_heid)

    new_edge_index = torch.cat(expanded_n2he_index, dim=1)
#     new_he2n_index = torch.cat(expanded_he2n_index, dim = 1)
#     new_edge_index = torch.cat([new_n2he_index, new_he2n_index], dim = 1)
    # sort the new_edge_index by first row. (node_ids)
    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order]

    return data


# functions for processing/checkning the edge_index
def get_HyperGCN_He_dict(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index)
    """
    For each he, clique-expansion. Note that we allow the weighted edge.
    Note that if node pair (vi,vj) is contained in both he1, he2, we will have (vi,vj) twice in edge_index. (weighted version CE)
    We default no self loops so far.
    """
# #     Construct a dictionary
#     He2V_List = []
# #     Sort edge_index according to he_id
#     _, sorted_idx = torch.sort(edge_index[1])
#     edge_index = edge_index[:,sorted_idx].type(torch.LongTensor)
#     current_heid = -1
#     for idx, he_id in enumerate(edge_index[1]):
#         if current_heid != he_id:
#             current_heid = he_id
#             if idx != 0 and len(he2v)>1: #drop original self loops
#                 He2V_List.append(he2v)
#             he2v = []
#         he2v.append(edge_index[0,idx].item())
# #     Remember to append the last he
#     if len(he2v)>1:
#         He2V_List.append(he2v)
# #     Now, turn He2V_List into a dictionary
    edge_index[1, :] = edge_index[1, :]-edge_index[1, :].min()
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        #         ipdb.set_trace()
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

#     for he_id, he in enumerate(He2V_List):
#         He_dict[he_id] = he

    return He_dict


def ConstructH(data):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
#     ipdb.set_trace()
    edge_index = np.array(data.edge_index)
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1])-np.min(edge_index[1])+1
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.
        cur_idx += 1

    data.edge_index = H
    return data


def ConstructH_HNHN(data):
    """
    Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
    """
    edge_index = np.array(data.edge_index)
    num_nodes = data.n_x[0]
    num_hyperedges = int(data.totedges)
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.
        cur_idx += 1

#     data.incident_mat = H
    return H


def generate_G_from_H(data):
    """
    This function generate the propagation matrix G for HGNN from incidence matrix H.
    Here we assume data.edge_index is already the incidence matrix H. (can be done by ConstructH())
    Adapted from HGNN github repo: https://github.com/iMoonLab/HGNN
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
#     ipdb.set_trace()
    H = data.edge_index
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
#     replace nan with 0. This is caused by isolated nodes
    DV2 = np.nan_to_num(DV2)
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
    G = DV2 * H * W * invDE * HT * DV2
    data.edge_index = torch.Tensor(G)
    return data


def generate_G_for_HNHN(data, args):
    """
    This function generate the propagation matrix G_V2E and G_E2V for HNHN from incidence matrix H.
    Here we assume data.edge_index is already the incidence matrix H. (can be done by ConstructH())

    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
#     ipdb.set_trace()
    H = data.edge_index
    alpha = args.HNHN_alpha
    beta = args.HNHN_beta
    H = np.array(H)

    # the degree of the node
    DV = np.sum(H, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    G_V2E = np.diag(DE**(-beta))@H.T@np.diag(DV**(beta))
    G_E2V = np.diag(DV**(-alpha))@H@np.diag(DE**(alpha))

#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
    data.G_V2E = torch.Tensor(G_V2E)
    data.G_E2V = torch.Tensor(G_E2V)
    return data


def generate_norm_HNHN(H, data, args):
    """
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
#     H = data.incident_mat
    alpha = args.HNHN_alpha
    beta = args.HNHN_beta
    H = np.array(H)

    # the degree of the node
    DV = np.sum(H, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    num_nodes = data.n_x[0]
    num_hyperedges = int(data.totedges)
    # alpha part
    D_e_alpha = DE ** alpha
    D_v_alpha = np.zeros(num_nodes)
    for i in range(num_nodes):
        # which edges this node is in
        he_list = np.where(H[i] == 1)[0]
        D_v_alpha[i] = np.sum(DE[he_list] ** alpha)

    # beta part
    D_v_beta = DV ** beta
    D_e_beta = np.zeros(num_hyperedges)
    for i in range(num_hyperedges):
        # which nodes are in this hyperedge
        node_list = np.where(H[:, i] == 1)[0]
        D_e_beta[i] = np.sum(DV[node_list] ** beta)

    D_v_alpha_inv = 1.0 / D_v_alpha
    D_v_alpha_inv[D_v_alpha_inv == float("inf")] = 0

    D_e_beta_inv = 1.0 / D_e_beta
    D_e_beta_inv[D_e_beta_inv == float("inf")] = 0

    data.D_e_alpha = torch.from_numpy(D_e_alpha).float()
    data.D_v_alpha_inv = torch.from_numpy(D_v_alpha_inv).float()
    data.D_v_beta = torch.from_numpy(D_v_beta).float()
    data.D_e_beta_inv = torch.from_numpy(D_e_beta_inv).float()

    return data


def ConstructV2V(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index)
    """
    For each he, clique-expansion. Note that we DONT allow duplicated edges.
    Instead, we record its corresponding weights.
    We default no self loops so far.
    """
# # Note that the method below for CE can be memory expensive!!!
#     new_edge_index = []
#     for he in np.unique(edge_index[1, :]):
#         nodes_in_he = edge_index[0, :][edge_index[1, :] == he]
#         if len(nodes_in_he) == 1:
#             continue #skip self loops
#         combs = combinations(nodes_in_he,2)
#         for comb in combs:
#             new_edge_index.append([comb[0],comb[1]])


#     new_edge_index, new_edge_weight = torch.tensor(new_edge_index).type(torch.LongTensor).unique(dim=0,return_counts=True)
#     data.edge_index = new_edge_index.transpose(0,1)
#     data.norm = new_edge_weight.type(torch.float)

# # Use the method below for better memory complexity
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1
            else:
                edge_weight_dict[comb] += 1

# # Now, translate dict to edge_index and norm
#
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1

    data.edge_index = torch.tensor(new_edge_index).type(torch.LongTensor)
    data.norm = torch.tensor(new_norm).type(torch.FloatTensor)
    return data


def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x[0]
    num_hyperedges = data.num_hyperedges[0]
    if not ((data.n_x[0]+data.num_hyperedges[0]-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data


def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.n_x[0]
    num_hyperedges = data.num_hyperedges[0]

    if not ((data.n_x[0] + data.num_hyperedges[0] - 1) == data.edge_index[1].max().item()):
        print('num_hyperedges does not match! 2')
        return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros(
        (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data


def norm_contruction(data, option='all_one', TYPE='V2E'):
    if TYPE == 'V2E':
        if option == 'all_one':
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == 'deg_half_sym':
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
            V_norm = Vdeg**(-1/2)
            E_norm = HEdeg**(-1/2)
            data.norm = V_norm[data.edge_index[0]] * \
                E_norm[data.edge_index[1]-cidx]

    elif TYPE == 'V2V':
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True)
    return data


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
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

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx


