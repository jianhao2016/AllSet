#! /bin/sh
#
# Copyright (C) 
#
# Distributed under terms of the MIT license.
#


method=$1
cuda=0


dataset_list=( house-committees-100 walmart-trips-100 )

for feature_noise in 0 0.2 0.4 0.6 0.8 1
do
    for lr in 0.01 0.001
    do
        for wd in 0 1e-5
        do
            for dname in ${dataset_list[*]} 
            do
                source run_one_model.sh $dname $method $lr $wd ${feature_noise} $cuda
            done
        done   
    done
done
