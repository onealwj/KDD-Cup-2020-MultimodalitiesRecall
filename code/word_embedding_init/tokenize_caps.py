import os
import nltk
from utils import find_unicode
import sys
import argparse


def tokenize(sentence,file_token):
    words = nltk.tokenize.word_tokenize(sentence.lower().decode('utf-8'))

    for i in words:
         i = find_unicode(i)
         file_token.write(str(i)+'\t')
         file_token.flush()
    file_token.write('\n')


def tokenize_main(data_path, data_name, save_path):

    fname1 = os.path.join(os.path.join(data_path,data_name),'valid_caps.txt')
    fname2 = os.path.join(os.path.join(data_path,data_name),'train_caps.txt')
    fnames=[fname1,fname2]

    save_path= os.path.join(save_path,'valid_train_caps.txt')
    file_token=open(save_path, 'w')
    ii=0
    for fname in fnames:
         question_path = fname
         print(question_path)
         questions = open(question_path,'r')

         question = questions.readline()
         while question:
             if ii%500==0:
                 print(ii, question)
             tokenize(question, file_token)
             question=questions.readline()
             ii+=1

    questions.close()
    file_token.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--data_name', default='kdd2020_caps')
    parser.add_argument('--save_path', default='../vocab/')
    opt = parser.parse_args()

    tokenize_main(opt.data_path,opt.data_name,opt.save_path)


