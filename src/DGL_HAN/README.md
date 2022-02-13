# This is a modified version of DGL's implementation of HAN.

The current version is tested with `dgl==0.7.1` 


## Preprocessing of dataset:
```
python print_datset_statistics.py
```

## Full batch training:
```
python main.py --dataset cora --cuda 0 --runs 20 --feature_noise 1
```

## Mini batch training:
```
python train_sampling.py --dataset cora --device cuda:0 --runs 20 --feature_noise 1
```
