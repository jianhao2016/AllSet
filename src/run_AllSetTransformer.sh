#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#


dataset_list=( cora citeseer pubmed coauthor_cora coauthor_dblp \
    NTU2012 ModelNet40 zoo Mushroom 20newsW100 \
    yelp house-committees-100 walmart-trips-100 )
lr=0.001
wd=0
cuda=0

runs=20
epochs=500

for dname in in ${dataset_list[*]} 
do
    if [ "$dname" = "cora" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 4 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "citeseer" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "pubmed" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_cora" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_dblp" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "zoo" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 64 \
            --Classifier_hidden 64 \
            --wd 0.00001 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr 0.01
        echo "Finished training on ${dname}"
    elif [ "$dname" = "20newsW100" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "Mushroom" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "NTU2012" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "ModelNet40" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "yelp" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 64 \
            --Classifier_hidden 64 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "house-committees-100" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 1.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 0.6"
    elif [ "$dname" = "walmart-trips-100" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 1.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 0.6"   
    fi
done


echo "Finished all training for AllSetTransformer!"
