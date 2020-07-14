import pickle
import os
import time
import shutil

import torch

import data
#from vocab import Vocabulary  # NOQA
from vocab import Vocabulary, deserialize_vocab
from model import VSRN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, nDCG5_t2i, nDCG5_t2i_rerank, nDCG5_t2i_atten_rerank
import logging
import sys
import argparse



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.05, type=float,
                        help='loss margin.')
    parser.add_argument('--temperature', default=14, type=int,
                        help='loss temperature.')
    parser.add_argument('--num_epochs', default=9, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=2048, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=4, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed.')
    parser.add_argument('--use_atten', action='store_true',
                        help='use_atten')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                         help='Attention softmax temperature.')
    parser.add_argument('--use_box', action='store_true',
                        help='use_box')
    parser.add_argument('--use_label', action='store_true',
                        help='use_label')
    parser.add_argument('--use_mmd', action='store_true',
                        help='use_mmd')
    parser.add_argument('--score_path', default='../user_data/score.npy', type=str)
 
    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    #tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    #vocab = pickle.load(open(os.path.join(
    #    opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    #vocab = deserialize_vocab(os.path.join(opt.vocab_path, 'kdd2020_caps_vocab_train_val_threshold2.json'))
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    stoppath = os.path.join(opt.vocab_path, 'stopwords.txt')
    f_stop = open(stoppath,'r')
    stops = f_stop.readlines()
    stopwords=[]
    for sw in stops:
        sw = sw.strip()#.encode('utf-8').decode('utf-8')
        stopwords.append(sw)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, stopwords, opt.batch_size, opt.workers, opt, True)

    # Construct the model
    model = VSRN(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

def validate(opt, val_loader, model):
     # compute the encoding for all the validation images and captions
     #img_embs, cap_embs, cap_lens = encode_data(
     with torch.no_grad():
         q_id_dict = encode_data(
             model, val_loader, opt.log_step, logging.info)

         start = time.time()
         if opt.use_atten:
            rerank_ndcg5 = nDCG5_t2i_atten_rerank(q_id_dict, None, opt)
         else:
             ndcg5 = nDCG5_t2i_rerank(q_id_dict, None, opt)
         end = time.time()
         print("calculate similarity time:", end-start)
     #logging.info("Text to image: %.5f" % ndcg5)

     #tb_logger.log_value('ndcg5', ndcg5, step=model.Eiters)



if __name__ == '__main__':
    main()
