# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:21:16 2018

@author: chinatan
"""
raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",              
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
##语料 是一个包含9个字符串的数组
stoplist=set('for a of the and to in'.split(' '))
texts=[[word for word in documents.lower().split() if word not in stoplist] for documents in raw_corpus]
##把文档分割成一个个单词

from collections import defaultdict
frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1
preprocessed_corpus=[[token for token in text if frequency[token]>1]for text in texts]
preprocessed_corpus
## 统计词频次  大于1的输出
## 在处理之前 语料中每一个单词关联一个唯一的ID
from gensim import corpora
dictionary=corpora.Dictionary(preprocessed_corpus)
print(dictionary)

##单词顺序
print (dictionary.token2id)

##对句子 向量化
new_doc="human computer interaction"
new_vec=dictionary.doc2bow(new_doc.lower().split())
print (new_vec)
##把原始的语料转化为一组向量
bow_corpus=[dictionary.doc2bow(text) for text in preprocessed_corpus]
bow_corpus
##tf-idf 模型把词袋在模型表达的向量转化为另一个向量空间
from gensim import models
tfidf=models.TfidfModel(bow_corpus)
string='system minors'
string_bow=dictionary.doc2bow(string.lower().split())
string_tfidf=tfidf[string_bow]
print (string_bow)
print (string_tfidf)
## 用自己的语料库进行模型训练，检查“system minors” 在语料中的权重