from __future__ import print_function
import os
import pickle

import torch
import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
from model import VSRN, order_sim, func_attention, cosine_similarity
from collections import OrderedDict
import math
import json
import pandas as pd
from rerank import re_ranking


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    max_n_region = 0
    q_id_dict = {}
    for i, (images, captions, images_lengths, lengths, images_mask, txt_masks, product_ids, query_ids, bbox) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
        max_n_region = max(max_n_region, max(images_lengths))
        for q_id in query_ids:
            q_id_dict[q_id] = []


    for i, (images, captions, images_lengths, lengths, images_mask, txt_masks, product_ids, query_ids, bbox) in enumerate(data_loader):
    #for i, (images, captions, lengths, ids, caption_labels, caption_masks) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, cap_len, img_mask, GCN_img_emd, class_scores = model.forward_emb(images, captions, lengths, images_mask, txt_masks, bbox,
                                             volatile=True)
        '''
        for j, q_id in enumerate(query_ids):
            p_id = product_ids[j]
            q_id_dict[q_id].append((p_id, img_emb[j], cap_emb[j]))
            #q_id_dict[q_id].append((p_id, img_emb[j], cap_emb[j], img_len[j], cap_len[j], img_mask[j]))
        '''

        for j, q_id in enumerate(query_ids):
            p_id = product_ids[j]
            q_id_dict[q_id].append((p_id, img_emb[j], cap_emb[j], cap_len[j], img_mask[j]))


        '''
        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()


        del images, captions
        '''
        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        #del images, captions
    return q_id_dict

def nDCG5(score_dict, answer):

    ndcg = 0
    tem = []
    for k, v in answer.items():
        key = int(k)
        score_id = score_dict[key][0:5]
        p_ids = []
        for e in score_id:
            p_ids.append(int(e[1]))

        dcg = 0
        for i, e in enumerate(p_ids):
            flag = False
            for g in v:
                if e == g:
                    flag = True
                    break
            if flag:
                rel = 1.
            else:
                rel = 0

            dcg += rel / math.log(i+2, 2)
        idcg = 0
        for i in range(min(5,len(v))):
           idcg += 1.0 / math.log(i+2, 2)
        tem.append((k, dcg / idcg))
        ndcg += dcg / idcg
    tem.sort(key=lambda x: x[1], reverse=True)
    return ndcg / len(answer) 

def nDCG5_t2i(q_id_dict, answer, opt):
    score_dict = {}
    for k, v in q_id_dict.items():
        score_dict[k] = []
    for k, v in q_id_dict.items():
        for e in v:
            p_id = e[0]
            im_emb = e[1].unsqueeze(0)
            cap_emb = e[2].unsqueeze(0)
            #im_len = e[3]
            #cap_len = e[4]
            #im_mask = e[5].unsqueeze(0)

            im = im_emb
            s = cap_emb
            row_sim = im.mm(s.t())
            #row_sim = cosine_similarity(cap_emb, weiContext, dim=2).unsqueeze(0)

            score_dict[k].append((row_sim.cpu().detach().numpy()[0][0], p_id))
        score_dict[k].sort(key=lambda x: x[0], reverse=True)


    if answer is None:
        save_result(score_dict)
        return None
    else:
        ndcg5 = nDCG5(score_dict, answer)

        return ndcg5


def nDCG5_t2i_atten(q_id_dict, answer, opt):
     score_dict = {}
     for k, v in q_id_dict.items():
         score_dict[k] = []
     for k, v in q_id_dict.items():
         for e in v:
             p_id = e[0]
             im_emb = e[1].unsqueeze(0)
             cap_emb = e[2].unsqueeze(0)
             cap_len = e[3]
             im_mask = e[4].unsqueeze(0)
             weiContext, attn = func_attention(cap_emb, im_emb, im_mask, opt, smooth=opt.lambda_softmax)
             row_sim = cosine_similarity(cap_emb, weiContext, dim=2).unsqueeze(0)
             row_sim = row_sim.mean(dim=1, keepdim=True)

             score_dict[k].append((row_sim.cpu().detach().numpy()[0][0], p_id))
         score_dict[k].sort(key=lambda x: x[0], reverse=True)

     if answer is None:
         save_result(score_dict)
         return None
     else:
         ndcg5 = nDCG5(score_dict, answer)

         return ndcg5


def nDCG5_t2i_rerank(q_id_dict, answer, opt):
    score_dict = {}
    for k, v in q_id_dict.items():
        score_dict[k] = []

    for k, v in q_id_dict.items():
        q_g = []
        g_g = []
        p_id_list = []
        for e in v:
            p_id = e[0]
            im_emb = e[1].unsqueeze(0)
            cap_emb = e[2].unsqueeze(0)
            #im_len = e[3]
            #cap_len = e[4]
            #im_mask = e[5].unsqueeze(0)

            im = im_emb
            s = cap_emb
            row_sim = im.mm(s.t())
            q_g.append(row_sim.cpu().detach().numpy()[0][0])
            g_g.append(im.squeeze(0).cpu().detach().numpy())
            p_id_list.append(p_id)

            #row_sim = cosine_similarity(cap_emb, weiContext, dim=2).unsqueeze(0)

            score_dict[k].append((row_sim.cpu().detach().numpy()[0][0], p_id))

        q_g = np.array([q_g], dtype=np.float32)
        q_g = 1 - (q_g + 1) /2.
        g_g = np.array(g_g, dtype=np.float32)
        g_g = np.matmul(g_g, g_g.T)
        g_g = 1 - (g_g + 1) /2.
        q_q = np.zeros((1,1), dtype=np.float32)
        a = re_ranking(q_g, q_q, g_g)
        a = a[0]
        score_dict[k] = [(a[i], p_id_list[i]) for i in range(a.shape[0])]

        score_dict[k].sort(key=lambda x: x[0], reverse=False)
        #score_dict[k].sort(key=lambda x: x[0], reverse=True)

    np.save(opt.score_path, score_dict)

    if answer is None:
        save_result(score_dict)
        return None
    else:
        ndcg5 = nDCG5(score_dict, answer)

        return ndcg5



def nDCG5_t2i_atten_rerank(q_id_dict, answer, opt):
    score_dict = {}
    for k, v in q_id_dict.items():
        score_dict[k] = []

    count = 0
    for k, v in q_id_dict.items():
        count += 1
        if count % 20 == 0:
            print(count, '/',len(q_id_dict))
        q_g = []
        g_g = []
        p_id_list = []
        mask_list = []
        for e in v:
            p_id = e[0]
            im_emb = e[1].unsqueeze(0)
            cap_emb = e[2].unsqueeze(0)
            cap_len = e[3]
            im_mask = e[4].unsqueeze(0)
            weiContext, attn = func_attention(cap_emb, im_emb, im_mask, opt, smooth=opt.lambda_softmax)
            row_sim = cosine_similarity(cap_emb, weiContext, dim=2).unsqueeze(0)
            row_sim = row_sim.mean(dim=1, keepdim=True)

            im = im_emb
            s = cap_emb
            q_g.append(row_sim.cpu().detach().numpy()[0][0])
            g_g.append(im.detach())
            p_id_list.append(p_id)
            mask_list.append(im_mask.detach())

            #row_sim = cosine_similarity(cap_emb, weiContext, dim=2).unsqueeze(0)

            score_dict[k].append((row_sim.cpu().detach().numpy()[0][0], p_id))

        q_g = np.array([q_g], dtype=np.float32)

        q_g = 1 - (q_g + 1) /2.
        g_g_score = np.ones((len(g_g),len(g_g)), dtype=np.float32)
        '''
        for i in range(1,len(g_g)): 
            for j in range(i): 
                a = g_g[i][0]
                b = g_g[j][0]
                #im_mask = mask_list[j]
                sim = torch.mm(a, b.t())
                row_sim = torch.mean(sim)
                #weiContext, attn = func_attention(a, b, im_mask, opt, smooth=opt.lambda_softmax)
                #row_sim = cosine_similarity(a, weiContext, dim=2).unsqueeze(0)
                #row_sim = row_sim.mean(dim=1, keepdim=False)[0]
                g_g_score[i,j] = row_sim
                g_g_score[j,i] = row_sim
        '''
        for i in range(1,len(g_g)): 
            for j in range(i): 
                e = v[0]
                cap_emb = e[2].unsqueeze(0)
         
                a = g_g[i]
                b = g_g[j]
                im_mask = mask_list[j]
                weiContext1, attn = func_attention(cap_emb, a, mask_list[i], opt, smooth=opt.lambda_softmax)
                weiContext2, attn = func_attention(cap_emb, b, im_mask, opt, smooth=opt.lambda_softmax)
                #weiContext, attn = func_attention(a, b, im_mask, opt, smooth=opt.lambda_softmax)
                #row_sim = cosine_similarity(a, weiContext, dim=2).unsqueeze(0)
                row_sim = cosine_similarity(weiContext1, weiContext2, dim=2).unsqueeze(0)
                row_sim = row_sim.mean(dim=1, keepdim=False)[0]
                g_g_score[i,j] = row_sim
                g_g_score[j,i] = row_sim
  
        g_g = g_g_score

        g_g = 1 - (g_g + 1) /2.
        q_q = np.zeros((1,1), dtype=np.float32)
        a = re_ranking(q_g, q_q, g_g)
        a = a[0]
        score_dict[k] = [(a[i], p_id_list[i]) for i in range(a.shape[0])]

        score_dict[k].sort(key=lambda x: x[0], reverse=False)
        #score_dict[k].sort(key=lambda x: x[0], reverse=True)
    np.save(opt.score_path, score_dict)
        

    if answer is None:
        save_result(score_dict)
        return None
    else:
        ndcg5 = nDCG5(score_dict, answer)

        return ndcg5





def save_result(score_dict):
    r = []
    for k,v in score_dict.items():
        score_id = score_dict[k][0:5]
        p_ids = [k]
        for e in score_id:
            p_ids.append(int(e[1]))
        r.append(p_ids)


    df = pd.DataFrame(r, columns=['query-id','product1','product2','product3','product4','product5'])
    df.to_csv('../prediction_result/submission.csv', sep=',', index=False)







def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # construct model
    model = VSRN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure=opt.measure,
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure=opt.measure,
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
