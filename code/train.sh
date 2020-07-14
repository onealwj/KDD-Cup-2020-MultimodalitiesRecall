#!/bin/bash 

#train six models using different seed
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/baseline_seed1_rerank6 --log_step 40 --seed 1 
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/baseline_seed12_rerank6 --log_step 40 --seed 12 
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/baseline_seed123_rerank6 --log_step 40 --seed 123 
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/baseline_seed1234_rerank6 --log_step 40 --seed 1234
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/baseline_seed12345_rerank6 --log_step 40 --seed 12345
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/baseline_seed123456_rerank6 --log_step 40 --seed 123456

#finetune 
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/fine16_baseline_seed1_rerank6 --log_step 40 --seed 1 --num_epochs 10 --resume ../user_data/baseline_seed1_rerank6/model_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/fine16_baseline_seed12_rerank6 --log_step 40 --seed 12 --num_epochs 10 --resume ../user_data/baseline_seed12_rerank6/model_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/fine16_baseline_seed123_rerank6 --log_step 40 --seed 123 --num_epochs 10 --resume ../user_data/baseline_seed123_rerank6/model_rerank_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/fine16_baseline_seed1234_rerank6 --log_step 40 --seed 1234 --num_epochs 10 --resume ../user_data/baseline_seed1234_rerank6/model_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/fine16_baseline_seed12345_rerank6 --log_step 40 --seed 12345 --num_epochs 10 --resume ../user_data/baseline_seed12345_rerank6/model_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python train.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/fine16_baseline_seed123456_rerank6 --log_step 40 --seed 1234567 --num_epochs 10 --resume ../user_data/baseline_seed123456_rerank6/model_best.pth.tar
