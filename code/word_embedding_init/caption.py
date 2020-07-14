# -*- coding: UTF-8 -*-
import pandas as pd
import csv
import os
import argparse


def get_valid_query(data_path, data_name):
    print('begin getting valid query')
    valid_path=os.path.join(os.path.join(data_path,'valid'),'valid.tsv')
    valid_data=pd.read_csv(valid_path, sep='\t')
    filename =os.path.join(os.path.join(data_path, data_name), 'valid_caps.txt')

    with open(filename,'w') as file:
        for i in range(0,len(valid_data['product_id'])):
            query =  valid_data['query'][i]
            if i % 1000 == 0:
                print(i, query)
            file.write(query+'\n')


def get_train_query(data_path, data_name):
    print('begin getting valid query')
    train_path=os.path.join(os.path.join(data_path,'train'),'train.tsv')
    train_data=pd.read_csv(train_path, sep='\t', chunksize=5, quoting=csv.QUOTE_NONE)#error_bad_lines=False)
    filename =os.path.join(os.path.join(data_path, data_name), 'train_caps.txt')
    i = 0
    n = 0
    file= open(filename,'w')
    for td in train_data:
        step = td.shape[0]
        for j in range(0, step):
          query = td['query'][i]
          file.write(str(query)+'\n')
          file.flush()
          if i % 1000 == 0:
            print(i, query)
          i += 1
        del td
        n += step


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--data_name', default='kdd2020_caps')

    opt = parser.parse_args()

    path = os.path.join(opt.data_path,opt.data_name)
    if not os.path.exists(path):
        os.makedirs(path)

    get_valid_query(opt.data_path,opt.data_name)
    get_train_query(opt.data_path,opt.data_name)
