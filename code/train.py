import pickle
import os
import time
import shutil
import torch

#import data
import data_oversampling as data
#import data_hardsampler_v1 as data
#import data_hardsampler as data
import data_hardsampler_finetune as data_finetune
from vocab import Vocabulary, deserialize_vocab
from model import VSRN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, nDCG5_t2i, nDCG5_t2i_rerank, nDCG5_t2i_atten, nDCG5_t2i_atten_rerank
import logging
import sys
import argparse
import random
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    parser.add_argument('--num_epochs', default=7, type=int,
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
    parser.add_argument('--use_box', action='store_true',
                        help='use_box')
    parser.add_argument('--use_label', action='store_true',
                        help='use_label')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                         help='Attention softmax temperature.')
    parser.add_argument('--use_mmd', action='store_true',
                        help='use_mmd')
    parser.add_argument('--score_path', default='../user_data/score.npy', type=str)

    opt = parser.parse_args()
    print(opt)

    set_seed(opt.seed)

    if not os.path.exists(opt.logger_name):
        os.mkdir(opt.logger_name)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

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
    if opt.resume:
        train_loader, val_loader = data_finetune.get_loaders(
            opt.data_name, vocab, stopwords, opt.batch_size, opt.workers, opt)
    else:
        train_loader, val_loader = data.get_loaders(
            opt.data_name, vocab, stopwords, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = VSRN(opt)

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = 4
            #start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' epoch {}"
                  .format(opt.resume, start_epoch))
            #validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    best_rerank_rsum = 0

    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train_loader.dataset.initial()
        best_rsum, best_rerank_rsum = train(opt, train_loader, model, epoch, val_loader, best_rsum, best_rerank_rsum)

        # evaluate on validation set
        rsum, rerank_rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        rerank_is_best = rerank_rsum > best_rerank_rsum
        best_rsum = max(rsum, best_rsum)
        best_rerank_rsum = max(rerank_rsum, best_rerank_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'best_rerank_rsum': best_rerank_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, rerank_is_best, prefix=opt.logger_name + '/')



def train(opt, train_loader, model, epoch, val_loader, best_rsum, best_rerank_rsum):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        #model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            # validate(opt, val_loader, model)

            # evaluate on validation set
            rsum, rerank_rsum= validate(opt, val_loader, model)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            rerank_is_best = rerank_rsum > best_rerank_rsum
            best_rsum = max(rsum, best_rsum)
            best_rerank_rsum = max(rerank_rsum, best_rerank_rsum)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'best_rerank_rsum': best_rerank_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, rerank_is_best, prefix=opt.logger_name + '/')


    return best_rsum, best_rerank_rsum

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    #img_embs, cap_embs = encode_data(
    #    model, val_loader, opt.log_step, logging.info)

    with torch.no_grad():
        q_id_dict = encode_data(
            model, val_loader, opt.log_step, logging.info)

        start = time.time()
        if opt.use_atten:
            #rerank_ndcg5 = nDCG5_t2i_atten_rerank(q_id_dict, val_loader.dataset.answer, opt)
            rerank_ndcg5 = -1
            org_ndcg5 = nDCG5_t2i_atten(q_id_dict, val_loader.dataset.answer, opt)
        else:
            rerank_ndcg5 = nDCG5_t2i_rerank(q_id_dict, val_loader.dataset.answer, opt)
            org_ndcg5 = nDCG5_t2i(q_id_dict, val_loader.dataset.answer, opt)
        end = time.time()
        print("calculate similarity time:", end-start)
    logging.info("Text to image: org:%.5f rerank:%.5f" % (org_ndcg5, rerank_ndcg5))


    return org_ndcg5, rerank_ndcg5

def save_checkpoint(state, is_best, rerank_is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    if rerank_is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_rerank_best.pth.tar')

'''
def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    if epoch < 4:
        lr = opt.learning_rate
    elif epoch < 10:
        lr = opt.learning_rate * 0.1 
    else:
        lr = opt.learning_rate * 0.01 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()
