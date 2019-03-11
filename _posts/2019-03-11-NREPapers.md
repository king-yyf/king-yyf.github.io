---
layout: post
title: Must-read papers on NRE
date: 2019-03-11 
tags: 论文清单 
---


NRE: Neural Relation Extraction.

Contributed by [Tianyu Gao](https://github.com/gaotianyu1350) and [Xu Han](https://github.com/THUCSTHanxu13).

We released [OpenNRE](https://github.com/thunlp/OpenNRE), an open-source framework for neural relation extraction. This repository provides several relation extraction methods and an easy-to-use training and testing framework.



## Survey Papers

1. **A Survey of Deep Learning Methods for Relation Extraction.**
   _Shantanu Kumar._
   2017.
   [paper](https://arxiv.org/pdf/1705.03645.pdf)

2. **Relation Extraction : A Survey.**
   _Sachin Pawara,b, Girish K. Palshikara, Pushpak Bhattacharyyab._
   2017.
   [paper](https://arxiv.org/pdf/1712.05191.pdf)



## Datasets

### Supervised Datasets

1. **ACE 2005 Dataset** [link](https://catalog.ldc.upenn.edu/LDC2006T06)

2. **SemEval-2010 Task 8 Dataset** [link](http://semeval2.fbk.eu/semeval2.php?location=tasks#T11)

### Distantly Supervised Datasets

1. **NYT Dataset** [link](http://iesl.cs.umass.edu/riedel/ecml/)

### Few-shot Datasets

1. **FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation**
   _Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, Maosong Sun_
   EMNLP 2018.
   [paper](http://aclweb.org/anthology/D18-1514)

     > We present a Few-Shot Relation Classification Dataset (FewRel), consisting of 70,000 sentences on 100 relations derived from Wikipedia and annotated by crowdworkers.

## Word Vector Tools

1. **Word2vec** [link](code.google.com/p/word2vec)

2. **GloVe** [link](https://nlp.stanford.edu/projects/glove/)



## Journal and Conference papers:

### Supervised Datasets

1. **SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals.**
   _Iris Hendrickx , Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid O ́ Se ́aghdha, Sebastian Pado ́, Marco Pennacchiotti, Lorenza Romano, Stan Szpakowicz._
   Workshop on Semantic Evaluations, ACL 2009
   [paper](http://delivery.acm.org/10.1145/1630000/1621986/p94-hendrickx.pdf?ip=133.130.111.179&id=1621986&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1536636784_f0b60686e8866c0c08f63436f3ed81eb)

   > This leads us to introduce a new task, which will be part of SemEval-2010: multi-way classification of mutually exclusive semantic relations between pairs of common nominals.

### Distantly Supervised Datasets and Training Methods

1. **Learning to Extract Relations from the Web using Minimal Supervision.**
  _Razvan C. Bunescu, Department of Computer Sciences._
  ACL 2007.
  [paper](http://www.aclweb.org/anthology/P07-1073)

    > We present a new approach to relation extraction that requires only a handful of training examples. Given a few pairs of named entities known to exhibit or not exhibit a particular relation, bags of sentences containing the pairs are extracted from the web.

2. **Distant Supervision for Relation Extraction without Labeled Data.**
  _Mike Mintz, Steven Bills, Rion Snow, Dan Jurafsky._
  ACL-IJCNLP 2009.
  [paper](https://www.aclweb.org/anthology/P09-1113)

    > Our experiments use Freebase, a large semantic database of several thousand relations, to provide distant supervision.

3. **Modeling Relations and Their Mentions without Labeled Text.**
  _Sebastian Riedel, Limin Yao, Andrew McCallum._
  ECML 2010.
  [paper](https://link.springer.com/content/pdf/10.1007%2F978-3-642-15939-8_10.pdf)

    > We present a novel approach to distant supervision that can alleviate this problem based on the following two ideas: First, we use a factor graph to explicitly model the decision whether two entities are related, and the decision whether this relation is mentioned in a given sentence; second, we apply constraint-driven semi-supervision to train this model without any knowledge about which sentences express the relations in our training KB.

4. **Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations.**
  _Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer, Daniel S. Weld._
  ACL-HLT 2011.
  [paper](http://www.aclweb.org/anthology/P11-1055)

    > This paper presents a novel approach for multi-instance learning with overlapping re- lations that combines a sentence-level extrac- tion model with a simple, corpus-level compo- nent for aggregating the individual facts.

### Embeddings

1. **Distributed Representations of Words and Phrases and their Compositionality.**		
   _Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean._
   NIPS 2013.
   [paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

   > In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling.

2. **GloVe: Global Vectors for Word Representation.**
   _Jeffrey Pennington, Richard Socher, Christopher D. Manning._
   EMNLP 2014.
   [paper](http://www.aclweb.org/anthology/D14-1162)

   > The result is a new global log-bilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods.	

### Neural Encoders

1. **Semantic Compositionality through Recursive Matrix-Vector Spaces.**
  _Richard Socher, Brody Huval, Christopher D. Manning, Andrew Y. Ng._
  EMNLP-CoNLL 2012.
  [paper](https://ai.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf)

    > We introduce a recursive neural network (RNN) model that learns compositional vector representations for phrases and sentences of arbitrary syntactic type and length.

2. **Convolution Neural Network for Relation Extraction**
   _Chunyang Liu, Wenbo Sun, Wenhan Chao, Wanxiang Che._
   ADMA 2013
   [paper](https://link.springer.com/chapter/10.1007/978-3-642-53917-6_21)

   > In this paper, we propose a novel convolution network, incorporating lexical features, applied to Relation Extraction. 

3. **Relation Classification via Convolutional Deep Neural Network.**
  _Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, Jun Zhao._
  COLING 2014.
  [paper](http://www.aclweb.org/anthology/C14-1220)

    > We exploit a convolutional deep neural network (DNN) to extract lexical and sentence level features. Our method takes all of the word tokens as input without complicated pre-processing.

4. **Classifying Relations by Ranking with Convolutional Neural Networks.**
  _C´ıcero Nogueira dos Santos, Bing Xiang, Bowen Zhou._
  ACL 2015.
  [paper](https://www.aclweb.org/anthology/P15-1061)

    > In this work we tackle the relation classification task using a convolutional neural network that performs classification by ranking (CR-CNN).

5. **Relation Extraction: Perspective from Convolutional Neural Networks.**
   _Thien Huu Nguyen, Ralph Grishman._
   NAACL-HLT 2015
   [paper](http://www.aclweb.org/anthology/W15-1506)	

   > Our model takes advantages of multiple window sizes for filters and pre-trained word embeddings as an initializer on a non-static architecture to improve the performance. We emphasize the relation extraction problem with an unbalanced corpus.		

6. **End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures.**
   _Makoto Miwa, Mohit Bansal._
   ACL 2016.
   [paper](https://www.aclweb.org/anthology/P16-1105)

   > Our recurrent neural network based model captures both word sequence and dependency tree substructure information by stacking bidirectional tree-structured LSTM-RNNs on bidirectional sequential LSTM-RNNs... We further encourage detection of entities during training and use of entity information in relation extraction via entity pre-training and scheduled sampling.

7. **A Walk-based Model on Entity Graphs for Relation Extraction.**
   _Fenia Christopoulou, Makoto Miwa, Sophia Ananiadou._
   ACL 2018.
   [paper](http://aclweb.org/anthology/P18-2014#page=6&zoom=100,0,313)

   > We present a novel graph-based neural network model for relation extraction. Our model treats multiple pairs in a sentence simultaneously and considers interactions among them. All the entities in a sentence are placed as nodes in a fully-connected graph structure.

### Denoising Methods

1. **Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks.**
  _Daojian Zeng, Kang Liu, Yubo Chen, Jun Zhao._
  EMNLP 2015.
  [paper](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)

    > We propose a novel model dubbed the Piecewise Convolutional Neural Networks (PCNNs) with multi-instance learning to address these two problems.

2. **Neural Relation Extraction with Selective Attention over Instances.**
  _Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun._
  ACL 2016.
  [paper](http://www.aclweb.org/anthology/P16-1200)

    > Distant supervision inevitably accompanies with the wrong labelling problem, and these noisy data will substantially hurt the performance of relation extraction. To alleviate this issue, we propose a sentence-level attention-based model for relation extraction.

3. **Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks.**
   _Xiaotian Jiang, Quan Wang, Peng Li, Bin Wang._
   COLING 2016.
   [paper](http://www.aclweb.org/anthology/C16-1139)

   > In this paper, we propose a multi-instance multi-label convolutional neural network for distantly supervised RE. It first relaxes the expressed-at-least-once assumption, and employs cross-sentence max-pooling so as to enable information sharing across different sentences.		

4. **Adversarial Training for Relation Extraction.**
  _Yi Wu, David Bamman, Stuart Russell._
  EMNLP 2017.
  [paper](http://www.aclweb.org/anthology/D17-1187)

    > Adversarial training is a mean of regularizing classification algorithms by generating adversarial noise to the training data. We apply adversarial training in relation extraction within the multi-instance multi-label learning framework.

5. **A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction.**
  _Tianyu Liu, Kexiang Wang, Baobao Chang, Zhifang Sui._
  EMNLP 2017.
  [paper](http://www.aclweb.org/anthology/D17-1189)

    > We introduce an entity-pair level denoise method which exploits semantic information from correctly labeled entity pairs to correct wrong labels dynamically during training.

6. **DSGAN: Generative Adversarial Training for Distant Supervision Relation Extraction.**
  _Pengda Qin, Weiran Xu, William Yang Wang._
  [paper](https://arxiv.org/pdf/1805.09929.pdf)

    > We introduce an adversarial learning framework, which we named DSGAN, to learn a sentence-level true-positive generator. Inspired by Generative Adversarial Networks, we regard the positive samples generated by the generator as the negative samples to train the discriminator.

7. **Reinforcement Learning for Relation Classification from Noisy Data.**
  _Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu._
  AAAI 2018.
  [paper](https://tianjun.me/static/essay_resources/RelationExtraction/Paper/AAAI2018Denoising.pdf)

    > We propose a novel model for relation classification at the sentence level from noisy data. The model has two modules: an instance selector and a relation classifier. The instance selector chooses high-quality sentences with reinforcement learning and feeds the selected sentences into the relation classifier, and the relation classifier makes sentence-level prediction and provides rewards to the instance selector.

8. **Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning.**
  _Pengda Qin, Weiran Xu, William Yang Wang._
    2018.
  [paper](https://arxiv.org/pdf/1805.09927.pdf)

    > We explore a deep reinforcement learning strategy to generate the false-positive indicator, where we automatically recognize false positives for each relation type without any supervised information.

### Extensions

1. **Neural Knowledge Acquisition via Mutual Attention between Knowledge Graph and Text**
  _Xu Han, Zhiyuan Liu, Maosong Sun._
  AAAI 2018.
  [paper](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2018_jointnre.pdf)

    > We propose a general joint representation learning framework for knowledge acquisition (KA) on two tasks, knowledge graph completion (KGC) and relation extraction (RE) from text. We propose an effective mutual attention between KGs and text. The recip- rocal attention mechanism enables us to highlight important features and perform better KGC and RE.

1. **Hierarchical Relation Extraction with Coarse-to-Fine Grained Attention**
   _Xu Han, Pengfei Yu, Zhiyuan Liu, Maosong Sun, Peng Li._
   EMNLP 2018.
   [paper](http://www.aclweb.org/anthology/D18-1247)

     > We aim to incorporate the hierarchical information of relations for distantly supervised relation extraction and propose a novel hierarchical attention scheme. The multiple layers of our hierarchical attention scheme provide coarse-to-fine granularity to better identify valid instances, which is especially effective for extracting those long-tail relations.

1. **Incorporating Relation Paths in Neural Relation Extraction**
   _Wenyuan Zeng, Yankai Lin, Zhiyuan Liu, Maosong Sun._
   EMNLP 2017.
   [paper](http://aclweb.org/anthology/D17-1186)

     > We build inference chains between two target entities via intermediate entities, and propose a path-based neural relation extraction model to encode the relational semantics from both direct sentences and inference chains. 

1. **RESIDE: Improving Distantly-Supervised Neural Relation Extractionusing Side Information**
   _Shikhar Vashishth, Rishabh Joshi, Sai Suman Prayaga, Chiranjib Bhattacharyya, Partha Talukdar._
   EMNLP 2018.
   [paper](https://aclweb.org/anthology/D18-1157)

     > In this paper, we propose RESIDE, a distantly-supervised neural relation extraction method which utilizes additional side  information from KBs for improved relation extraction. It uses entity type and relation alias information for imposing soft constraints while predicting relations.
