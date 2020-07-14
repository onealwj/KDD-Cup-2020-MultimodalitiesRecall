import gensim
from gensim.models import word2vec
import os
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='../vocab/')
    opt = parser.parse_args()

    f_caps_all = os.path.join(opt.save_path,'valid_train_caps.txt')
    sentences_all = word2vec.LineSentence(f_caps_all)

    model = word2vec.Word2Vec(sentences_all, sg=1, hs=1, min_count=0, window=5, size=300)
    save_name_t = os.path.join(opt.save_path,'w2v_300d.txt')
    model.wv.save_word2vec_format(save_name_t)

    #save_name_m = '../data/train/w2v_300d_tail_'+str(i)+'.model'
    #model.save(save_name_m)


