#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#


dname=$1
method=$2
lr=0.001
wd=0
MLP_hidden=$3
Classifier_hidden=$4
feature_noise=$5
cuda=0

runs=10
epochs=500

if [ "$method" = "MLP" ]; then
    echo =============
    echo ">>>> Model MLP, Dataset: ${dname}"
    python train.py \
        --method MLP \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --feature_noise $feature_noise \
        --cuda $cuda \
        --lr $lr

elif [ "$method" = "AllDeepSets" ]; then
    echo =============
    echo ">>>> Model AllDeepSets, Dataset: ${dname}"
    python train.py \
        --method AllDeepSets \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --feature_noise $feature_noise \
        --runs $runs \
        --cuda $cuda \
        --lr $lr

elif [ "$method" = "AllSetTransformer" ]; then
    for heads in 1 4 8
    do
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}, head: ${heads}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise $feature_noise \
            --heads $heads \
            --Classifier_num_layers 1 \
            --MLP_hidden $MLP_hidden \
            --Classifier_hidden $Classifier_hidden \
            --wd $wd \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
    done
        
elif [ "$method" = "CEGCN" ]; then
    echo =============
    echo ">>>>  Model:CEGCN, Dataset: ${dname}"
    python train.py \
        --method CEGCN \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --cuda $cuda \
        --feature_noise $feature_noise \
        --lr $lr

elif [ "$method" = "CEGAT" ]; then
    for heads in 1 4 8
    do
        echo =============
        echo ">>>>  Model:CEGAT, Dataset: ${dname}"
        python train.py \
            --method CEGAT \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --Classifier_num_layers 1 \
            --MLP_hidden $MLP_hidden \
            --Classifier_hidden $Classifier_hidden \
            --heads $heads \
            --wd $wd \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --feature_noise $feature_noise \
            --lr $lr
    done
    
elif [ "$method" = "HyperGCN" ]; then
    echo =============
    echo ">>>>  Model:HyperGCN (mediator,fast), Dataset: ${dname}"
    python train.py \
        --method HyperGCN \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --HyperGCN_mediators \
        --HyperGCN_fast \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --feature_noise $feature_noise \
        --cuda $cuda \
        --lr $lr

    echo =============
    echo ">>>>  Model:HyperGCN (mediator,fast,no_self_loops), Dataset: ${dname}"
    python train.py \
        --method HyperGCN \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --HyperGCN_mediators \
        --HyperGCN_fast \
        --add_self_loop \
        --wd $wd \
        --feature_noise $feature_noise \
        --epochs $epochs \
        --cuda $cuda \
        --runs $runs \
        --lr $lr

elif [ "$method" = "HGNN" ]; then
    echo =============
    echo ">>>>  Model:HGNN (with sym-HCHA), Dataset: ${dname}"
    python train.py \
        --method HGNN \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --HCHA_symdegnorm \
        --wd $wd \
        --epochs $epochs \
        --feature_noise $feature_noise \
        --runs $runs \
        --cuda $cuda \
        --lr $lr
        
elif [ "$method" = "HNHN" ]; then
    echo =============
    echo ">>>>  Model:HNHN, Dataset: ${dname}"
    python train.py \
        --method HNHN \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --cuda $cuda \
        --feature_noise $feature_noise \
        --lr $lr
        
elif [ "$method" = "HCHA" ]; then
    echo =============
    echo ">>>>  Model:HCHA (asym deg norm), Dataset: ${dname}"
    python train.py \
        --method HCHA \
        --dname $dname \
        --All_num_layers 1 \
        --MLP_num_layers 2 \
        --Classifier_num_layers 1 \
        --MLP_hidden $MLP_hidden \
        --Classifier_hidden $Classifier_hidden \
        --wd $wd \
        --epochs $epochs \
        --runs $runs \
        --cuda $cuda \
        --feature_noise $feature_noise \
        --lr $lr

elif [ "$method" = "UniGCNII" ]; then
    echo =============
    echo ">>>>  Model:UniGCNII, Dataset: ${dname}"
    for heads in 1 4 8
    do
        python train.py \
            --method UniGCNII \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --Classifier_num_layers 1 \
            --MLP_hidden $MLP_hidden \
            --Classifier_hidden $Classifier_hidden \
            --heads $heads \
            --wd $wd \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --feature_noise $feature_noise \
            --lr $lr
    done
fi

echo "Finished training ${method} on ${dname}"
