#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#


method=$1
cuda=0

# dname=$1
# method=$2
lr=0.001
wd=0
# MLP_hidden=$3
# Classifier_hidden=$4
feature_noise=0.6

dataset_list=( cora citeseer pubmed coauthor_cora coauthor_dblp \
    NTU2012 ModelNet40 zoo Mushroom 20newsW100 \
    yelp house-committees-100 walmart-trips-100 )

method_list=( AllDeepSets AllSetTransformer MLP \
    UniGCNII HyperGCN CEGCN CEGAT HGNN HNHN HCHA )

for MLP_hidden in 64 128 256 512
do
    for Classifier_hidden in 64 128 256
    do
        for dname in ${dataset_list[*]} 
        do
            for method in ${method_list[*]}
            do
                source run_one_model_cuda1.sh $dname $method $MLP_hidden $Classifier_hidden $feature_noise
            done
        done
    done   
done
