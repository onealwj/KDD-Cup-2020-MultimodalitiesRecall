#!/bin/bash 

CUDA_VISIBLE_DEVICES=$1 python test.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/test --log_step 40 --seed 123 --score_path ../user_data/score1.npy --resume ../user_data/fine16_baseline_seed1_rerank6/model_rerank_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python test.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/test --log_step 40 --seed 123 --score_path ../user_data/score12.npy --resume ../user_data/fine16_baseline_seed12_rerank6/model_rerank_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python test.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/test --log_step 40 --seed 123 --score_path ../user_data/score123.npy --resume ../user_data/fine16_baseline_seed123_rerank6/model_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python test.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/test --log_step 40 --seed 123 --score_path ../user_data/score1234.npy --resume ../user_data/fine16_baseline_seed1234_rerank6/model_rerank_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python test.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/test --log_step 40 --seed 123 --score_path ../user_data/score12345.npy --resume ../user_data/fine16_baseline_seed12345_rerank6/model_rerank_best.pth.tar
CUDA_VISIBLE_DEVICES=$1 python test.py --data_path ../data/ --data_name kdd2020_precomp --logger_name ../user_data/test --log_step 40 --seed 123 --score_path ../user_data/score123456.npy --resume ../user_data/fine16_baseline_seed123456_rerank6/model_rerank_best.pth.tar

python ensemble_v1.py ../user_data/score1.npy ../user_data/score12.npy ../user_data/score123.npy ../user_data/score1234.npy ../user_data/score12345.npy ../user_data/score123456.npy
