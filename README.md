# KDD CUP 2020: Multimodalities Recall
### Team: WST
## Introduction
Our intuitive idea is to learn to align queries with their target images. We develop a model following the work of “Visual Semantic Reasoning for Image-Text Matching (VSRN)”[1]. According to the aim of the task and the characteristics of the provided data, we have made several improvements on the architecture of VSRN, we apply a cross-modal cosine loss, which is inspired by the work of [3-5], and we use a re-ranking technique [2] in post processing. Our model reach the score of 0.8151 on valid dataset, and 0.7872 on testB.

## Requirement
- Python (3.6.9)
- NumPy (1.17.4)
- PyTorch (1.1.0)
- pandas (0.25.3)
- torchvision
-  Punkt Sentence Tokenizer (3.4.5):
```python
import nltk
nltk.download()
> d punkt
```

## Environment
We train our models on a single TITAN Xp GPU, CUDA 10.1, and CUDNN 7.6.4

## Prediction
The trained models can be downloaded from Baidu Driver
address: [https://pan.baidu.com/s/1SbF6NKbMVFlytBOWxCqIuQ](https://pan.baidu.com/s/1SbF6NKbMVFlytBOWxCqIuQ)
password: 8fnd
- **mkdir data/ user_data/ prediction_result/ external_resources/**
- Download all zip files and unzip to folder **user_data/** 
- Run **./main.sh ${GPU}**
- Get prediction result file submission.csv from folder **prediction_result/**

## Train
Run **./train.sh ${GPU}**

### Vocabulary and Word Embeddings
We have provided the vocabulary file and pre-trained Word2vec embeddings in 'code/vocab/' directory. Alternatively, you can also produce them according to the following steps:
#### Requirements
- Python(2.7.14)
- gensim(3.8.1)

#### Vocabulary
```
cd word_embedding_init
python caption.py --data_path ../../data --data_name kdd2020_caps
python vocab.py --data_path ../../data --data_name kdd2020_caps
```

#### Word embedding
```
cd word_embedding_init
python tokenize_caps.py --data_path ../../data --data_name kdd2020_caps --save_path ../vocab
python train_word2vec.py --save_path ../vocab
python init_word2vec_embedding.py --data_path ../vocab --data_name kdd2020_caps
```

## Reference
[1] Li K, Zhang Y, Li K, et al. Visual Semantic Reasoning for Image-Text Matching[C]. international conference on computer vision, 2019: 4654-4662.
[2] Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-Reciprocal Encoding[C]. computer vision and pattern recognition, 2017: 3652-3661.
[3] Li S, Xiao T, Li H, et al. Identity-Aware Textual-Visual Matching with Latent Co-attention[C]. international conference on computer vision, 2017: 1908-1917.
[4] Wang H, Wang Y, Zhou Z, et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition[C]. computer vision and pattern recognition, 2018: 5265-5274.
[5] Deng J, Guo J, Xue N, et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition[J]. arXiv: Computer Vision and Pattern Recognition, 2018

