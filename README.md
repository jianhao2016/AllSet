# AllSet

This is the repo for our paper: [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/forum?id=hpBTIv2uy_E). We prepared all codes and a subset of datasets used in our experiments.

All codes and script are in the folder `src`, and a subset of raw data are provided in folder `data`. To run the experiments, please go the the `src` folder first. 

## Enviroment requirement:
This repo is tested with the following enviroment, higher version of torch PyG may also be compatible. 

First let's setup a conda enviroment
```
conda create -n "AllSet" python=3.7
conda activate AllSet
```

Then install pytorch and PyG packages with specific version.
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-sparse==0.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-cluster==1.5.2 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric==1.6.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
```
Finally, install some relative packages

```
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib
```

## Generate dataset from raw data.

To generate PyG or DGL dataset for training, please create the following three folders:
```
p2root: './data/pyg_data/hypergraph_dataset_updated/'
p2raw: './data/AllSet_all_raw_data/'
p2dgl_data: './data/dgl_data_raw/'
```

And then unzip the raw data zip file into `p2raw`.


## Run one single experiment with one model with specified lr and wd: 
```
source run_one_model.sh [dataset] [method] [MLP_hidden_dim] [Classifier_hidden_dim] [feature noise level]
```
Note that for HAN, please check the readme file in `./src/DGL_HAN/`.

## To reproduce the results in Table 2(with the processed raw data)
```
source run_all_experiments.sh [method]
```
Notably, if you just want to reproduce the performance of AllSetTransformer in Table 2 without hyperparameter tuning, you can just run:
```
source run_AllSetTransformer.sh
```

**Remark:** We do not fix the random seed in our code so the results might be slightly different. If you find a huge discrepancy, please open an issue. Also, our table is obtained by complete grid search of hyperparameters and report the test accuracy based on best validation accuracy. So for results that have large variance (i.e. zoo), merely running it for 20 runs with best hyperparameters may still give accuracy different from Table 2. In this case we suggests to either increase the number of runs or follow our grid search exactly.

## Issues
If you have any problem about our code, please open an issue **and** @ us (or send us an email) in case the notification doesn't work. Our email can be found in the paper.

## Citation
If you use our code or data in your work, please cite our paper:
```
@inproceedings{
chien2022you,
title={You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks},
author={Eli Chien and Chao Pan and Jianhao Peng and Olgica Milenkovic},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=hpBTIv2uy_E}
}
```

