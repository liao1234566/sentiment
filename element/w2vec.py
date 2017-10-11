# -*- coding: utf-8 -*-
import codecs
import multiprocessing

from gensim.models.word2vec import Word2Vec, LineSentence


# 数据训练得到词向量
def train(trainPath, modelPath):
    model = Word2Vec(LineSentence(trainPath), size=128, window=5, min_count=5,workers=multiprocessing.cpu_count())
    model.save(modelPath)
    print("训练完成！")


# 测试
def test(modelPath):
    model = Word2Vec.load(modelPath)
    #近义词
    #print(model.most_similar('孩子'))
    #相似度
    # print(model.similarity('埃及','孩子'))
    vec = model['幼儿园']
    print(vec)

if __name__ == '__main__':
    trainPath = '../../data/element/corpus.txt'
    modelPath = '../../data/element/w2vec.model'
    # 1.训练
#     train(trainPath, modelPath)
    # 2.测试
    test(modelPath)
