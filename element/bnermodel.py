#-*- coding: utf-8 -*-
'''
Created on 2017年6月20日

@author: Administrator
'''

import codecs
import json

from gensim.models import Word2Vec
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.datasets.tests.test_samples_generator import test_make_multilabel_classification_return_sequences
from sklearn.model_selection import train_test_split

import eletags, elepre, eleload
from ner.element import viterbi
import numpy as np


def train(cwsInfo, cwsData, modelPath, weightPath,w2vecPath):

    (initProb, tranProb), (vocab, indexVocab) = cwsInfo
    (X, y) = cwsData
    train_X, test_X, train_y, test_y = train_test_split(X, y , train_size=0.9, random_state=1)
    
#     (train_X,train_y)=cwsData
#     (test_X,test_y)=testData
    
    train_X = np.array(train_X)#转换weigh矩阵
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    outputDims = len(eletags.corpus_tags)
    Y_train = np_utils.to_categorical(train_y, outputDims)#标签用one-hot表示
    Y_test = np_utils.to_categorical(test_y, outputDims)#标签用one-hot表示
    batchSize = 128
    vocabSize = len(vocab) + 1
    wordDims = 100
    maxlen = 7
    hiddenDims = 128

    w2vModel = Word2Vec.load(w2vecPath)
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    embeddingWeights = np.zeros((vocabSize + 1, embeddingDim))
    for word, index in vocab.items():
        if word in w2vModel:
            e = w2vModel[word]
        else:
            e = embeddingUnknown
        embeddingWeights[index, :] = e

    #LSTM
    model = Sequential()
    #输入层
    model.add(Embedding(input_dim = vocabSize + 1,output_dim = embeddingDim, input_length = maxlen, mask_zero = True, weights = [embeddingWeights]))
    #隐层
    model.add(Bidirectional(LSTM(output_dim = hiddenDims,return_sequences=True)))
#     model.add(TimeDistributed(Dense(outputDims)))
#     model.add(Dropout(0.5))#Dropout 层用于防止过拟合
    model.add(Bidirectional(LSTM(output_dim = hiddenDims,return_sequences=False)))
    model.add(Dropout(0.5))#Dropout 层用于防止过拟合
    #输出层
    model.add(Dense(outputDims))#全链接层
    model.add(Activation('softmax'))#激活层对一个层的输出施加激活函数
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=["accuracy","precision", "recall", "fmeasure"])

    model.fit(train_X, Y_train, batch_size = batchSize, nb_epoch = 10,verbose=1, validation_data = (test_X,Y_test))

#     plot(model, to_file='model.png',show_shapes=True)
    j = model.to_json()
    fd = open(modelPath, 'w')
    fd.write(j)
    fd.close()
    model.save_weights(weightPath)
    return model

#加载模型
def loadModel(modelPath, weightPath):
    fd = open(modelPath, 'r')
    j = fd.read()
    fd.close()
    model = model_from_json(j)
    model.load_weights(weightPath)
    return model

#单句测试
def testModel(sent,nerModelPath,nerWeightPath,nerInfoPath):
    nerInfo=eleload.loadNerInfo(nerInfoPath)
    model=loadModel(nerModelPath, nerWeightPath)
    result=nerSent(sent, model, nerInfo)
    print(result)
    
# 根据输入得到标注推断
def nerSent(sent, model, nerInfo):
    (initProb, tranProb), (vocab, indexVocab) = nerInfo
    vec = elepre.sent2vec(sent, vocab, ctxWindows = 7)
    vec = np.array(vec)
    classes=model.predict_classes(vec)
    ss=''
    for i,t in enumerate(classes):
        tag=eletags.corpus_tags[t]
        ss+=sent[i]+'/'+tag+' '

#     probs = model.predict_proba(vec)
#     prob, path = viterbi.viterbi(vec, nertags.corpus_tags, initProb, tranProb, probs.transpose())
#     ss = ''
#     for i, t in enumerate(path):
#         tag=nertags.corpus_tags[t]
#         ss+=sent[i]+'/'+tag+' '
    return ss

def nerSentList(sents, model, nerInfo):
    (initProb, tranProb), (vocab, indexVocab) = nerInfo
    results=[]
    for sent in sents:
        vec = elepre.sent2vec(sent, vocab, ctxWindows = 7)
        vec = np.array(vec)
#         classes=model.predict_classes(vec)
#         ss=''
#         for i,t in enumerate(classes):
#             tag=nertags.corpus_tags[t]
#             ss+=sent[i]+'/'+tag+' '
#         results.append(ss)
        probs = model.predict_proba(vec)
        prob, path = viterbi.viterbi(vec, eletags.corpus_tags, initProb, tranProb, probs.transpose())
        ss = ''
        for i, t in enumerate(path):
            tag=eletags.corpus_tags[t]
            ss+=sent[i]+'/'+tag+' '
        results.append(ss)
    return results

#训练
def trainModel(nerModelPath,nerWeightPath,nerInfoPath,nerDataPath):
    print('-------开始加载词典------')
    nerInfo=eleload.loadNerInfo(nerInfoPath)
    nerData=eleload.loadNerData(nerDataPath)
#     testData=nerload.loadNerData(testDataPath)
    print('------载入词典完成-------')
    
    print('------------开始训练-----------')
    train(nerInfo, nerData, nerModelPath, nerWeightPath, nerW2vecPath)
    print('------------训练完毕---------')

#多句测试
def testList(sents,nerModelPath,nerWeightPath,nerInfoPath):
    nerInfo=eleload.loadNerInfo(nerInfoPath)
    model=loadModel(nerModelPath, nerWeightPath)
    result=nerSentList(sents, model, nerInfo)
    return result

if __name__ == '__main__':
    basePath="../../"
    nerModelPath=basePath+'data/element/ele_blstm20.model'
    nerWeightPath=basePath+'data/element/ele_blstm20.weight'
    nerW2vecPath=basePath+'data/element/c2vec.model'
    nerInfoPath=basePath+'data/element/eleInfo_win-7.info'
    nerDataPath=basePath+'data/element/eleData_win-7.data'
#     testDataPath=basePath+'data/data/testData.data'
# 1.训练
#     trainModel(nerModelPath, nerWeightPath, nerW2vecPath, nerInfoPath, nerDataPath)
# 2.测试
    sent="坚决抵制蒙牛牛奶,坚决抵制！宁愿去喝妇炎洁。"
    testModel(sent, nerModelPath, nerWeightPath, nerInfoPath)
    pass