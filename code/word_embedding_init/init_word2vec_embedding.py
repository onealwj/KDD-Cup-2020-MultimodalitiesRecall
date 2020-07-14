"""
This code is modified by Peng Ying from Linjie Li's repository.
https://github.com/linjieli222/VQA_ReGAT
"""
from __future__ import print_function
import os
import sys
import json
import numpy as np
from utils import find_unicode
import argparse

def create_w2v_embedding_init(idx2word, w2v_file):
    word2emb = {}
    with open(w2v_file, 'r') as f:
        entries = f.readlines()
    entry = entries[0].split(' ')
    #print(entries[0].split(' '))
    num_words = int(entry[0])
    emb_dim = int(entry[1])
    print('there are', num_words, 'words', 'the embedding dim is', emb_dim)

    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)
    #print(weights.shape)

    for entry in entries[1:]:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)

    count=0
    count_=0
    for idx, word in idx2word.items():
        if word not in word2emb:
            count+=1
            #print(count,'not found word', word)
            flag = 0

            ws = word.split('-')
            if '' in ws:
              for iw in range(len(ws)):
                if ws[iw] != '':
                    w = ws[iw]
                    if w in word2emb:
                        weights[int(idx)] = word2emb[w]
                        #print(word, w)
                        count_+=1
                        flag = 1
                        break
            if flag == 0:
                w = find_unicode(word)
                if w in word2emb:
                    weights[int(idx)] = word2emb[w]
                    #print(word, w)
                    count_+=1
                    flag = 1

            if flag == 0:
                weights[int(idx)] = np.random.uniform(-0.1, 0.1)

            '''
            if (flag ==0) and (word not in ['<start>','<unk>','<end>','<pad>']):
               weights[int(idx)] = np.random.uniform(-0.1, 0.1)
            '''
            continue
        weights[int(idx)] = word2emb[word]
    #print(count, count_)
    return weights, word2emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../vocab/')
    parser.add_argument('--data_name', default='kdd2020_caps')

    opt = parser.parse_args()

    vocab_path = os.path.join(opt.data_path,opt.data_name+'_vocab.json')
    dictionary_f = open(vocab_path,'r')
    #print(vocab_path)
    dictionary = json.load(dictionary_f)
    idx2word = dictionary['idx2word']

    #w2v_file = '/home/ubuntu/cvs/pengying/KDD2020/word2vec/word2vec300d_all3.txt'
    w2v_file = os.path.join(opt.data_path,'w2v_300d.txt')

    weights, word2emb = create_w2v_embedding_init(idx2word, w2v_file)

    save_path = os.path.join(opt.data_path,'word2vec300d_init_threshold4.npy')
    np.save(save_path, weights)

