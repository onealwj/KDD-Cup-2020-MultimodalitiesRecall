import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from GCN_lib.Rs_GCN import Rs_GCN

import torch.optim as optim

def rbf(x, y, gamma):
  """RBF kernel K(x,y) """
  pdist = torch.norm(x[:, None] - y, dim=2, p=2)
  return torch.exp(-gamma * pdist)


def l2norm(X, dim=1, mask=None, eps=1e-8):
     """L2-normalize columns of X
     """
     if mask is None:
         norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
         X = torch.div(X, norm)
     else:
         m = mask.unsqueeze(2)
         m = 1-m
         X = X + m
         norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
         X = torch.div(X, norm)
         X  = X * mask.unsqueeze(2)

     return X



class EncoderImagePrecompAttn(nn.Module):

    def __init__(self, img_dim, embed_size, inter_channels=2048, use_atten=False, use_box=False, use_label=False):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.use_atten = use_atten
        self.use_box = use_box
        self.use_label = use_label
        if self.use_box:
            self.fc_box = nn.Linear(6, embed_size)
        if self.use_label:
            self.fc_class = nn.Linear(embed_size, 33)
        #self.fc = nn.Linear(img_dim, inter_channels)

        #self.init_weights()


        # GSR
        self.img_rnn = nn.GRU(inter_channels, embed_size, 1, batch_first=True, bidirectional=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)
        self.Rs_GCN_2 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)
        self.Rs_GCN_3 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)
        self.Rs_GCN_4 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)
 


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, masks, bbox):
        """Extract image feature vectors."""


        fc_img_emd = images
        #fc_img_emd = self.fc(images)
        fc_img_emd = fc_img_emd * masks.unsqueeze(2)

        if self.use_box:
            w = bbox[:,:,3] - bbox[:,:,1]
            h = bbox[:,:,2] - bbox[:,:,0]
            cat_bbox = torch.cat((bbox, w.unsqueeze(2), h.unsqueeze(2)), dim=-1)
            fc_box_emd = self.fc_box(cat_bbox)
            fc_box_emd = fc_box_emd * masks.unsqueeze(2)
            #fc_img_emd = fc_img_emd + fc_box_emd 
            fc_img_emd = fc_box_emd 

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd, masks)
        GCN_img_emd = GCN_img_emd* masks.unsqueeze(2).permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd, masks)
        GCN_img_emd1 = GCN_img_emd* masks.unsqueeze(2).permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd1, masks)
        GCN_img_emd = GCN_img_emd* masks.unsqueeze(2).permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd, masks)

        GCN_img_emd = GCN_img_emd + GCN_img_emd1 



        #GCN_img_emd = self.se(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = GCN_img_emd* masks.unsqueeze(2)

        lenghs = torch.sum(masks, dim=1)
        l, indeces = torch.sort(lenghs,0,True)
        GCN_img_emd = torch.index_select(GCN_img_emd, 0, indeces)
        packed = pack_padded_sequence(GCN_img_emd, l.cpu().numpy().astype(np.int32), batch_first=True)
        rnn_img, hidden_state = self.img_rnn(packed)

        if self.use_atten:
            padded = pad_packed_sequence(rnn_img, batch_first=True)
            img_emb, img_len = padded
            img_emb = (img_emb[:,:,:int(img_emb.size(2)/2)] + img_emb[:,:,int(img_emb.size(2)/2):])/2.

            leng, ind = torch.sort(indeces,0)
            features = torch.index_select(img_emb, 0, ind)
            features = l2norm(features, dim=-1, mask=masks)
        else:
 
            features = (hidden_state[0]+hidden_state[1])/2.
            leng, ind = torch.sort(indeces,0)
            features = torch.index_select(features, 0, ind)
            features = l2norm(features, dim=-1)

        if self.use_label:
            GCN_img_emd = GCN_img_emd.reshape(GCN_img_emd.size()[0]*GCN_img_emd.size()[1], -1)
            class_scores = self.fc_class(GCN_img_emd)
            class_scores = class_scores * (masks.reshape(-1).unsqueeze(1))
        else:
            class_scores = None
        return features, GCN_img_emd, class_scores



    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompAttn, self).load_state_dict(new_state)


class EncoderTextPrecompAttn(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size=2048, inter_channels=2048, use_atten=False, word_vec='./vocab/word2vec300d_init_threshold4.npy'):
        super(EncoderTextPrecompAttn, self).__init__()
        self.use_atten = use_atten
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, word_dim)
        weight_init = torch.from_numpy(np.load(word_vec))
        self.embed.weight.data[:vocab_size] = weight_init

        self.fc = nn.Linear(word_dim, inter_channels)

        self.init_weights()


        # GSR
        self.img_rnn = nn.GRU(inter_channels, embed_size, 1, batch_first=True, bidirectional=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)
        self.Rs_GCN_2 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)
        self.Rs_GCN_3 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)
        self.Rs_GCN_4 = Rs_GCN(in_channels=inter_channels, inter_channels=inter_channels)



    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        #self.embed.weight.data.uniform_(-0.1, 0.1)
 

    def forward(self, x, masks):
        """Extract image feature vectors."""


        x = self.embed(x)
        fc_img_emd = self.fc(x)
        
        fc_img_emd = fc_img_emd * masks.unsqueeze(2)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd, masks)
        GCN_img_emd = GCN_img_emd* masks.unsqueeze(2).permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd, masks)

        GCN_img_emd = GCN_img_emd* masks.unsqueeze(2).permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd, masks)
        GCN_img_emd = GCN_img_emd* masks.unsqueeze(2).permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd, masks)


        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = GCN_img_emd* masks.unsqueeze(2)

        lenghs = torch.sum(masks, dim=1)
        l, indeces = torch.sort(lenghs,0,True)
        GCN_img_emd = torch.index_select(GCN_img_emd, 0, indeces)
        packed = pack_padded_sequence(GCN_img_emd, l.cpu().numpy().astype(np.int32), batch_first=True)
        rnn_img, hidden_state = self.img_rnn(packed)

        if self.use_atten:
            padded = pad_packed_sequence(rnn_img, batch_first=True)
            img_emb, img_len = padded
            if self.training:
                img_emb = (img_emb[:,:,:int(img_emb.size(2)/2)] + img_emb[:,:,int(img_emb.size(2)/2):])/2.
            else:
                img_emb = img_emb[:,:,:int(img_emb.size(2)/2)]

            leng, ind = torch.sort(indeces,0)
            features = torch.index_select(img_emb, 0, ind)
            features = l2norm(features, dim=-1, mask=masks)
        else:
            if self.training:
                features = (hidden_state[0]+hidden_state[1])/2.
            else:
                features = hidden_state[1]
            leng, ind = torch.sort(indeces,0)
            features = torch.index_select(features, 0, ind)
            features = l2norm(features, dim=-1)

        return features 


    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderTextPrecompAttn, self).load_state_dict(new_state)



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0.05, temperature=14, use_atten=False, use_mmd=False, use_label=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.sim = cosine_sim
        self.use_atten = use_atten
        self.use_label = use_label
        self.opt = opt

        self.pvse_mmd_loss = use_mmd

    def mmd_rbf_loss(self, x, y, gamma=None):
        if gamma is None:
            gamma = 1./x.size(-1)
        loss = rbf(x, x, gamma) - 2 * rbf(x, y, gamma) + rbf(y, y, gamma)
        return loss.mean()

    def forward(self, im, s, s_l, im_mask, query_id, class_scores, class_labels):
        if self.use_atten:
            scores = xattn_score_t2i(im, s, s_l, im_mask, self.opt)
        else:
            scores = self.sim(im, s)



        loss_function = nn.CrossEntropyLoss()
        temp = []
        labels = []
        for i in range(scores.size(0)):
            labels.append(i)
            if query_id[i] in temp:
                labels[-1] = -100
            temp.append(query_id[i])

        m = torch.eye(scores.size(0)).cuda()*self.margin
        scores_m = scores - m
        labels = Variable(torch.from_numpy(np.array(labels))).cuda()
        loss = loss_function(scores_m*self.temperature, labels)

        if False:
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)

            cost_s = (0.15 + scores - d1).clamp(min=0)
            cost_im = (0.15 + scores - d2).clamp(min=0)

            mask = torch.eye(scores.size(0)) > .5
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.cuda()
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)

            for i in range(scores.size(0)):
                for j in range(scores.size(0)):
                    if i != j:
                        if query_id[i] == query_id[j]:
                            cost_s[i,j] = 0
                            cost_im[i,j] = 0

            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

            #cost_s = cost_s.mean(dim=1)
            #cost_im = cost_im.mean(dim=0)

            _loss = cost_s.sum() + cost_im.sum()
            loss += _loss*0.02


        if self.use_label:
            class_labels = class_labels.reshape(-1)
            class_loss_func = nn.CrossEntropyLoss()
            class_labels = Variable(class_labels.long()).cuda()
            class_loss = class_loss_func(class_scores, class_labels)
            loss += class_loss * 0.5

        if self.pvse_mmd_loss:
            s_pvse=s #---
            mmd_weight = 1
            mmd_loss = self.mmd_rbf_loss(im.view(-1, im.size(-1)), s_pvse.view(-1, s_pvse.size(-1)), gamma=0.5)
            loss += mmd_weight * mmd_loss

        return loss



class VSRN(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.use_atten = opt.use_atten
        self.use_box = opt.use_box
        self.use_mmd = opt.use_mmd
        self.use_label = opt.use_label

        self.img_enc = EncoderImagePrecompAttn(opt.img_dim, opt.embed_size, use_atten=self.use_atten, use_box=self.use_box, use_label=self.use_label)
        self.txt_enc = EncoderTextPrecompAttn(opt.vocab_size, opt.word_dim, opt.embed_size, use_atten=self.use_atten)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt, margin=opt.margin,
                                         temperature=opt.temperature,
                                         use_atten=self.use_atten,
                                         use_mmd=self.use_mmd, 
                                         use_label=self.use_label
                                         )

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, im_masks, txt_masks, bbox, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            im_masks = im_masks.cuda()
            txt_masks = txt_masks.cuda()
            bbox = bbox.cuda()

        # Forward

        cap_emb = self.txt_enc(captions, txt_masks)
        img_emb, GCN_img_emd, class_scores = self.img_enc(images, im_masks, bbox)
        return img_emb, cap_emb, lengths, im_masks, GCN_img_emd, class_scores
        #return img_emb, cap_emb, GCN_img_emd

    #def forward_loss(self, img_emb, cap_emb, query_id, **kwargs):
    def forward_loss(self, img_emb, cap_emb, cap_len, images_mask, query_id, class_scores, class_labels, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        #loss = self.criterion(img_emb, cap_emb, query_id)
        loss = self.criterion(img_emb, cap_emb, cap_len, images_mask, query_id, class_scores, class_labels)
        self.logger.update('Le_retrieval', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, caption_masks, images_lengths, images_masks, query_id, query, num_boxes, boxes, class_labels, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        #img_emb, cap_emb, GCN_img_emd = self.forward_emb(images, captions, lengths, images_masks, caption_masks, boxes)
        img_emb, cap_emb, cap_lens, im_masks, GCN_img_emd, class_scores = self.forward_emb(images, captions, lengths, images_masks, caption_masks, boxes)


        # calcualte captioning loss
        self.optimizer.zero_grad()

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        #loss = self.forward_loss(img_emb, cap_emb, query_id)
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, im_masks, query_id, class_scores, class_labels)

        self.logger.update('Le', loss.item(), img_emb.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

def func_attention(query, context, img_mask, opt, smooth, eps=1e-8):
     """
     query: (n_context, queryL, d)
     context: (n_context, sourceL, d)
     """
     batch_size_q, queryL = query.size(0), query.size(1)
     batch_size, sourceL = context.size(0), context.size(1)


     # Get attention
     # --> (batch, d, queryL)
     queryT = torch.transpose(query, 1, 2)

     # (batch, sourceL, d)(batch, d, queryL)
     # --> (batch, sourceL, queryL)
     attn = torch.bmm(context, queryT)
     attn = nn.LeakyReLU(0.1)(attn)
     attn = l2norm(attn, 2, img_mask)
     # --> (batch, queryL, sourceL)
     attn = torch.transpose(attn, 1, 2).contiguous()
     # --> (batch*queryL, sourceL)
     attn = attn.view(batch_size*queryL, sourceL)
     mask_repeat = img_mask.unsqueeze(1)
     mask_repeat = mask_repeat.repeat(1, queryL, 1)
     mask_repeat = mask_repeat.view(batch_size*queryL, sourceL)
     attn = softmax(attn*smooth, mask_repeat)
     #attn = nn.Softmax()(attn*smooth)
     # --> (batch, queryL, sourceL)
     attn = attn.view(batch_size, queryL, sourceL)
     # --> (batch, sourceL, queryL)
     attnT = torch.transpose(attn, 1, 2).contiguous()

     # --> (batch, d, sourceL)
     contextT = torch.transpose(context, 1, 2)
     # (batch x d x sourceL)(batch x sourceL x queryL)
     # --> (batch, d, queryL)
     weightedContext = torch.bmm(contextT, attnT)
     # --> (batch, queryL, d)
     weightedContext = torch.transpose(weightedContext, 1, 2)

     return weightedContext, attnT

def softmax(x, mask):
     m = torch.max(x, 1, keepdim=True)[0]
     y = torch.exp(x - m)
     #y = torch.exp(x)
     y = y * mask
     s = torch.sum(y, 1, keepdim=True)
     y = torch.div(y, s)
     return y

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
     """Returns cosine similarity between x1 and x2, computed along dim."""
     w12 = torch.sum(x1 * x2, dim)
     w1 = torch.norm(x1, 2, dim)
     w2 = torch.norm(x2, 2, dim)
     return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def xattn_score_t2i(images, captions, cap_lens, img_mask, opt):
     """
     Images: (n_image, n_regions, d) matrix of images
     Captions: (n_caption, max_n_word, d) matrix of captions
     CapLens: (n_caption) array of caption lengths
     """
     similarities = []
     n_image = images.size(0)
     n_caption = captions.size(0)
     for i in range(n_caption):
         # Get the i-th text description
         n_word = cap_lens[i]
         cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
         # --> (n_image, n_word, d)
         cap_i_expand = cap_i.repeat(n_image, 1, 1)
         """
             word(query): (n_image, n_word, d)
             image(context): (n_image, n_regions, d)
             weiContext: (n_image, n_word, d)
             attn: (n_image, n_region, n_word)
         """
         weiContext, attn = func_attention(cap_i_expand, images, img_mask, opt, smooth=opt.lambda_softmax)
         cap_i_expand = cap_i_expand.contiguous()
         weiContext = weiContext.contiguous()
         # (n_image, n_word)
         row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
         row_sim = row_sim.mean(dim=1, keepdim=True)
         similarities.append(row_sim)

     # (n_image, n_caption)
     similarities = torch.cat(similarities, 1)

     return similarities


