import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import json as jsonmod
import pandas as pd
import csv
import base64
import json
import random

class TrainDataset(data.Dataset):

    def __init__(self, data_path, data_split, vocab, stopwords, opt):
        self.vocab = vocab
        self.loc = data_path + '/'
        self.length = 3000000
        self.data_split = data_split
        self.nraws = 20000
        self.shuffle = True
        self.stopwords=stopwords

    def initial(self):
        self.reader = pd.read_csv(self.loc+'%s/train.tsv' % self.data_split, sep='\t', quoting=csv.QUOTE_NONE, iterator=True)
        self.samples = list()
        self.samples_new = list()
        self.samples = self.reader.get_chunk(self.nraws)
        class_label = {}
        for i in range(len(self.samples)):
            data = self.samples.iloc[i]
            query_id = data['query_id']
            if query_id not in class_label.keys():
                class_label[query_id] = 1
            elif query_id in class_label.keys():
                class_label[query_id] = class_label[query_id]+1
        for i in range(len(self.samples)):
            data = self.samples.iloc[i]
            query_id = data['query_id']
            if class_label[query_id] <= 10:
                self.samples_new.append(self.samples.iloc[i])
            if class_label[query_id] > 10 and class_label[query_id]<=50:
                if random.random()>0.4:
                    self.samples_new.append(self.samples.iloc[i])
            if class_label[query_id] > 50 and class_label[query_id]<=100:
                if random.random()>0.5:
                    self.samples_new.append(self.samples.iloc[i])
            if class_label[query_id] > 100 and class_label[query_id]<=250:
                if random.random()>0.6:
                    self.samples_new.append(self.samples.iloc[i])
            if class_label[query_id] > 250 and class_label[query_id]<=500:
                if random.random()>0.8:
                    self.samples_new.append(self.samples.iloc[i])
            if class_label[query_id] > 500:
                if random.random()>0.9:
                    self.samples_new.append(self.samples.iloc[i])
        self.samples = self.samples_new 
        del self.samples_new
        self.current_sample_num = len(self.samples) 
        self.num_load = 0
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.index)

    def __getitem__(self, index):

        idx = self.index[0]
        data = self.samples[idx]
        self.index = self.index[1:]
        self.current_sample_num-=1

        if self.current_sample_num<=0:
        # all the samples in the memory have been used, need to get the new samples
            del self.samples
            self.samples = self.reader.get_chunk(self.nraws)
            self.samples_new = list()
            class_label = {}
            for i in range(len(self.samples)):
                data = self.samples.iloc[i]
                query_id = data['query_id']
                if query_id not in class_label.keys():
                    class_label[query_id] = 1
                elif query_id in class_label.keys():
                    class_label[query_id] = class_label[query_id]+1
            for i in range(len(self.samples)):
                data = self.samples.iloc[i]
                query_id = data['query_id']
                if class_label[query_id] <= 10:
                    self.samples_new.append(self.samples.iloc[i])
                if class_label[query_id] > 10 and class_label[query_id]<= 50:
                    if random.random()>0.4:
                        self.samples_new.append(self.samples.iloc[i])
                if class_label[query_id] > 50 and class_label[query_id]<= 100:
                    if random.random()>0.5:
                        self.samples_new.append(self.samples.iloc[i])
                if class_label[query_id] > 100 and class_label[query_id]<= 250:
                    if random.random()>0.6:
                        self.samples_new.append(self.samples.iloc[i])
                if class_label[query_id] > 250 and class_label[query_id]<= 500:
                    if random.random()>0.8:
                        self.samples_new.append(self.samples.iloc[i])
                if class_label[query_id] > 500:
                    if random.random()>0.9:
                        self.samples_new.append(self.samples.iloc[i])
            self.samples = self.samples_new 
            del self.samples_new
            self.current_sample_num = len(self.samples) 
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.index)


        num_boxes = int(data['num_boxes'])
        image = np.frombuffer(base64.b64decode(data['features']), dtype=np.float32).reshape(num_boxes, 2048)
        boxes =  np.frombuffer(base64.b64decode(data['boxes']), dtype=np.float32).reshape(num_boxes, 4).copy()
        class_labels = np.frombuffer(base64.b64decode(data['class_labels']), dtype=np.int64).reshape(num_boxes, 1)
        caption = data['query']
        query = data['query']
        query_id = int(data['query_id'])
        image_h = int(data['image_h'])
        image_w = int(data['image_w'])
        boxes[:,0] = boxes[:,0] / float(image_h)
        boxes[:,2] = boxes[:,2] / float(image_h)
        boxes[:,1] = boxes[:,1] / float(image_w)
        boxes[:,3] = boxes[:,3] / float(image_h)

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().encode('utf-8').decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        #caption.extend([vocab(token) for token in tokens])
        '''
        for token in tokens:
            if token not in self.stopwords:
                caption.append(vocab(token))
        '''

        #---py #stop and change to unk
        change_flag = 0
        rr=random.random()
        #caption_mid = []
        if rr>=0.8:
            for token in tokens:
                rrr=random.random()
                if (rrr>=0.8) and (change_flag==0):
                    change_flag=1
                    token = '<unk>'
                if token not in self.stopwords:
                    caption.append(vocab(token))
        else:
            for token in tokens:
                if token not in self.stopwords:
                    caption.append(vocab(token))
         #---py

        #caption_mid = caption_mid + caption_mid
        #caption = caption + caption_mid
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        image = torch.Tensor(image)
        boxes = torch.Tensor(boxes)
        class_labels = torch.Tensor(class_labels)

        #s = (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])
        #_, index = torch.sort(s,0)
        _, index = torch.sort(boxes[:,0],0)
        image = image[index,:]
        boxes = boxes[index,:]
        class_labels = class_labels[index,:]

        gf = torch.mean(image, dim=0, keepdim=True)
        image = torch.cat([image, gf], dim=0)
        return image, target, query_id, query, num_boxes, boxes, class_labels

    def __len__(self):
        return self.length

class ValidDataset(data.Dataset):

    def __init__(self, data_path, data_split, vocab, stopwords):
        self.vocab = vocab
        self.loc = data_path + '/'
        self.data_split = data_split
        self.reader = pd.read_csv(self.loc+'%s/valid.tsv' % self.data_split, sep='\t', quoting=csv.QUOTE_NONE)
        self.f = open(self.loc+'%s/valid_answer.json' % self.data_split)
        self.answer = json.load(self.f)
        self.length = self.reader.shape[0]
        self.stopwords = stopwords

    def __getitem__(self, index):

        pair = self.reader.loc[index]

        product_id = pair['product_id']
        query_id = pair['query_id']
        num_boxes = pair['num_boxes']
        image = np.frombuffer(base64.b64decode(pair['features']), dtype=np.float32).reshape(num_boxes, 2048)
        caption = pair['query']

        vocab = self.vocab

        boxes =  np.frombuffer(base64.b64decode(pair['boxes']), dtype=np.float32).reshape(num_boxes, 4).copy()
        image_h = int(pair['image_h'])
        image_w = int(pair['image_w'])
        boxes[:,0] = boxes[:,0] / float(image_h)
        boxes[:,2] = boxes[:,2] / float(image_h)
        boxes[:,1] = boxes[:,1] / float(image_w)
        boxes[:,3] = boxes[:,3] / float(image_h)

        boxes = torch.Tensor(boxes)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().encode('utf-8').decode('utf-8'))
        caption = []
        #caption_mid = []
        caption.append(vocab('<start>'))
        for token in tokens:
            if token not in self.stopwords:
                caption.append(vocab(token))
        #caption.extend([vocab(token) for token in tokens])
        #caption_mid = caption_mid + caption_mid
        #caption = caption + caption_mid
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        image = torch.Tensor(image)

        #s = (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])
        #_, index = torch.sort(s,0)
        _, index = torch.sort(boxes[:,0],0)
        image = image[index,:]
        boxes = boxes[index,:]
        gf = torch.mean(image, dim=0, keepdim=True)
        image = torch.cat([image, gf], dim=0)
        return image, target, product_id, query_id, boxes 

    def __len__(self):
        return self.length

class TestDataset(data.Dataset):

    def __init__(self, data_path, data_split, vocab, stopwords):
        self.vocab = vocab
        self.loc = data_path + '/'
        self.data_split = data_split
        self.reader = pd.read_csv(self.loc+'%s/testA.tsv' % self.data_split, sep='\t', quoting=csv.QUOTE_NONE)
        self.length = self.reader.shape[0]
        self.stopwords = stopwords

    def __getitem__(self, index):

        pair = self.reader.loc[index]

        product_id = pair['product_id']
        query_id = pair['query_id']
        num_boxes = pair['num_boxes']
        image = np.frombuffer(base64.b64decode(pair['features']), dtype=np.float32).reshape(num_boxes, 2048)
        caption = pair['query']

        vocab = self.vocab

        boxes =  np.frombuffer(base64.b64decode(pair['boxes']), dtype=np.float32).reshape(num_boxes, 4).copy()
        image_h = int(pair['image_h'])
        image_w = int(pair['image_w'])
        boxes[:,0] = boxes[:,0] / float(image_h)
        boxes[:,2] = boxes[:,2] / float(image_h)
        boxes[:,1] = boxes[:,1] / float(image_w)
        boxes[:,3] = boxes[:,3] / float(image_h)

        boxes = torch.Tensor(boxes)


        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().encode('utf-8').decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        for token in tokens:
            if token not in self.stopwords:
                caption.append(vocab(token)) 
        #caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        image = torch.Tensor(image)

        _, index = torch.sort(boxes[:,0],0)
        image = image[index,:]
        boxes = boxes[index,:]
        gf = torch.mean(image, dim=0, keepdim=True)
        image = torch.cat([image, gf], dim=0)
        return image, target, product_id, query_id, boxes

    def __len__(self):
        return self.length



def train_collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, query_id, query, num_boxes, boxes, class_labels = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images_lengths = [image.shape[0] for image in images]
    images_tensor = torch.zeros(len(images), max(images_lengths), 2048)
    images_masks = torch.zeros(len(images), max(images_lengths))
    boxes_tensor = torch.zeros(len(images), max(images_lengths), 4)
    class_labels_tensor = torch.ones(len(images), max(images_lengths), 1)*(-100)
    for i, image in enumerate(images):
        end = images_lengths[i]
        images_tensor[i, :end,:] = image[:,:]
        images_masks[i, :end] = 1
        boxes_tensor[i, :end-1,:] = boxes[i][:,:]
        class_labels_tensor[i, :end-1,:] = class_labels[i][:,:]

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    txt_masks = torch.zeros(len(captions), max(lengths))
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        txt_masks[i, :end] = 1

    return images_tensor, targets, lengths, txt_masks, images_lengths, images_masks, query_id, query, num_boxes, boxes_tensor, class_labels_tensor

def valid_collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, product_ids, query_ids, boxes = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images_lengths = [image.shape[0] for image in images]
    images_tensor = torch.zeros(len(images), max(images_lengths), 2048)
    images_masks = torch.zeros(len(images), max(images_lengths))
    boxes_tensor = torch.zeros(len(images), max(images_lengths), 4)
    for i, image in enumerate(images):
        end = images_lengths[i]
        images_tensor[i, :end,:] = image[:,:]
        images_masks[i, :end] = 1
        boxes_tensor[i, :end-1,:] = boxes[i][:,:]
        #images_tensor[i, :end,:] = image[:end,:]
    #images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    txt_masks = torch.zeros(len(captions), max(lengths))
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        txt_masks[i, :end] = 1

    return images_tensor, targets, images_lengths, lengths, images_masks, txt_masks, product_ids, query_ids, boxes_tensor





def get_train_loader(data_path, data_split, vocab, stopwords, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = TrainDataset(data_path, data_split, vocab, stopwords, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=train_collate_fn)
    return data_loader

def get_valid_loader(data_path, data_split, vocab, stopwords, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = ValidDataset(data_path, data_split, vocab, stopwords)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=valid_collate_fn)
    return data_loader


def get_loaders(data_name, vocab, stopwords, batch_size, workers, opt, only_test=False):
    dpath = opt.data_path
    if not only_test:
        train_loader = get_train_loader(dpath, 'train', vocab, stopwords, opt,
                                      batch_size, True, workers)
        val_loader = get_valid_loader(dpath, 'valid', vocab, stopwords, opt,
                                    batch_size, False, workers)
    else:
        train_loader = None
        val_loader = get_test_loader(dpath, 'testA', vocab, stopwords, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader

def get_test_loader(data_path, data_split, vocab, stopwords, opt, batch_size=100,
                       shuffle=True, num_workers=2):

    dset = TestDataset(data_path, data_split, vocab, stopwords)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=valid_collate_fn)
    return data_loader


