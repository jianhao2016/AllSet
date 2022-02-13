import time
import torch
import ipdb
import numpy as np

from sklearn.metrics import f1_score
from pprint import pprint

from utils import load_data, EarlyStopping, get_binary_mask

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

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
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, _, _, _, _, _, _ = load_data(args['dataset'], feature_noise = args['feature_noise'])

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    num_nodes = features.shape[0]

    history_acc, history_micro_f1, history_marco_f1 = [], [], []
    run_time_list = []
    for run in range(args['runs']):
        # Get new split.
        start_time = time.time()

        new_split = rand_train_test_idx(labels, args['train_prop'], args['valid_prop'])
        train_idx, val_idx, test_idx = new_split['train'], new_split['valid'], new_split['test']

        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)

        if hasattr(torch, 'BoolTensor'):
            train_mask = train_mask.bool()
            val_mask = val_mask.bool()
            test_mask = test_mask.bool()

        train_mask = train_mask.to(args['device'])
        val_mask = val_mask.to(args['device'])
        test_mask = test_mask.to(args['device'])
        # pprint({
        #     'train': train_mask.sum().item() / num_nodes,
        #     'val': val_mask.sum().item() / num_nodes,
        #     'test': test_mask.sum().item() / num_nodes
        # })

        if args['hetero']:
            from model_hetero import HAN
            model = HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout']).to(args['device'])
            g = g.to(args['device'])
        else:
            from model import HAN
            model = HAN(num_meta_paths=len(g),
                        in_size=features.shape[1],
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        num_heads=args['num_heads'],
                        dropout=args['dropout']).to(args['device'])
            g = [graph.to(args['device']) for graph in g]

        stopper = EarlyStopping(patience=args['patience'])
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                     weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):
            model.train()
            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
            val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
            early_stop = stopper.step(val_loss.data.item(), val_acc, model)

            # print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
            #       'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            #     epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

            if early_stop:
                break


        stopper.load_checkpoint(model)
        test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
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
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')

    parser.add_argument('--dataset', default = 'ACM')
    parser.add_argument('--runs',type = int, default = 20)
    parser.add_argument('--cuda',type = int, default = 0)
    parser.add_argument('--feature_noise', type = float, default = 1)
    parser.add_argument('--train_prop', type = float, default = 0.5)
    parser.add_argument('--valid_prop', type = float, default = 0.25)

    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
