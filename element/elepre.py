#-*- coding: utf-8 -*-
'''
Created on 2017年1月7日

@author: Administrator
'''
import codecs
import json

import h5py

from eletags import corpus_tags, ctxWindows
from ner.element import eleload


# 保存命名实体训练数据字典和概率
def saveNerInfo(path, nerInfo):
    print('save cws info to %s'%path)
    fd = open(path, 'w',-1,'utf-8')
    (initProb, tranProb), (vocab, indexVocab) = nerInfo
    j = json.dumps((initProb, tranProb))
    fd.write(j + '\n')
    for char in vocab:
        fd.write(char + '\t' + str(vocab[char]) + '\n')
    fd.close()

# 保存分词训练输入样本'
def saveNerData(path, cwsData):
    print('save cws data to %s'%path)
    #采用hdf5保存大矩阵效率最高
    fd = h5py.File(path,'w')
    (X, y) = cwsData
    fd.create_dataset('X', data = X)
    fd.create_dataset('y', data = y)
    fd.close()
def sent2vec2(sent, vocab, ctxWindows = 5):
    charVec = []
    for char in sent:
        if char in vocab:
            charVec.append(vocab[char])
        else:
            print(char)
            charVec.append(vocab['retain-unknown'])
    #首尾padding
    num = len(charVec)
    pad = int((ctxWindows - 1)/2)
    for i in range(pad):
        charVec.insert(0, vocab['retain-padding'] )
        charVec.append(vocab['retain-padding'] )
    X = []
    for i in range(num):
        X.append(charVec[i:i + ctxWindows])
    return X

def sent2vec(sent, vocab, ctxWindows = 5):
    chars = []
    for char in sent:
        chars.append(char)
    return sent2vec2(chars, vocab, ctxWindows = ctxWindows)

# 文档转向量
def doc2vec(fname, vocab):
    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    #样本集
    X = []
    y = []

    #标注统计信息
    tagSize = len(corpus_tags)
    tagCnt = [0 for i in range(tagSize)]
    tagTranCnt = [[0 for i in range(tagSize)] for j in range(tagSize)]

    #遍历行
    for line in lines:
        #按空格分割
        words = line.strip('\n').strip('\r').split('\t')
        #每行的分词信息
        chars = []
        tags = []
        #遍历词
        for word in words:
            res=word.split('/')
            if len(res)<2:continue
            word=res[0] 
            tag=res[1]
            if tag=='Product' or tag=='Feature': #如果是产品或极性
                if len(word)>1:#如果大于等于两个字
                    #首字
                    chars.append(word[0])
                    tags.append(corpus_tags.index(tag+'-B'))
                    #非首字
                    for char in word[1:len(word)]:
                        chars.append(char)
                        tags.append(corpus_tags.index(tag+'-I'))
                else:#如果单个字
                    chars.append(word)
                    tags.append(corpus_tags.index(tag+'-B'))
            else:#如果不是产品或极性
                for char in word:
                    chars.append(char)
                    tags.append(corpus_tags.index(tag))
        #字向量表示
        lineVecX = sent2vec2(chars, vocab, ctxWindows = ctxWindows)

        #统计标注信息
        lineVecY = []
        lastTag = -1
        for tag in tags:
            #向量
            lineVecY.append(tag)
            #统计tag频次
            tagCnt[tag] += 1
            #统计tag转移频次
            if lastTag != -1:
                tagTranCnt[lastTag][tag] += 1
            #暂存上一次的tag
            lastTag = tag

        X.extend(lineVecX)
        y.extend(lineVecY)

    #字总频次
    charCnt = sum(tagCnt)
    #转移总频次
    tranCnt = sum([sum(tag) for tag in tagTranCnt])
    #tag初始概率
    initProb = []
    for i in range(tagSize):
        initProb.append(tagCnt[i]/float(charCnt))
    #tag转移概率
    tranProb = []
    for i in range(tagSize):
        p = []
        for j in range(tagSize):
            p.append(tagTranCnt[i][j]/float(tranCnt))
        tranProb.append(p)

    return X, y, initProb, tranProb

def vocabAddChar(vocab, indexVocab, index, char):
    if char not in vocab:
        vocab[char] = index
        indexVocab.append(char)
        index += 1
    return index

# 构造字典和索引
def genVocab(fname, delimiters = [' ', '\n']):
    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()
    vocab = {}
    indexVocab = []
    #遍历所有行
    index = 0
    for line in lines:
        words = line.strip().split('\t')
        if len(words)<= 0: continue
        #遍历所有词
        for word in words:
            res=word.split('/')
            if len(res)<2:continue
            word=res[0]
            for char in word:
                if char not in delimiters:#如果为分隔符则无需加入字典
                    index=vocabAddChar(vocab, indexVocab, index, char)
    #加入未登陆新词和填充词
    vocab['retain-unknown'] = len(vocab)
    vocab['retain-padding'] = len(vocab)
    indexVocab.append('retain-unknown')
    indexVocab.append('retain-padding')
    #返回字典与索引
    return vocab, indexVocab

def load(fname):
    vocab, indexVocab = genVocab(fname)
    X, y, initProb, tranProb = doc2vec(fname, vocab)
    print (len(X), len(y), len(vocab), len(indexVocab))
    return (X, y), (initProb, tranProb), (vocab, indexVocab)

if __name__ == '__main__':
    infoPath='../../data/element/eleInfo_win-7.info'
    dataPath='../../data/element/eleData_win-7.data'

#1.抽取信息并保存
    fname = '../../data/element/element.txt'
    (X, y), (initProb, tranProb), (vocab, indexVocab) = load(fname)
    nerInfo = (initProb, tranProb), (vocab, indexVocab)
    nerData = (X, y)
    saveNerInfo(infoPath,nerInfo)
    saveNerData(dataPath, nerData)

# 2.测试一下加载后的信息
    (initProb, tranProb), (vocab, indexVocab) = eleload.loadNerInfo(infoPath)
    (X, y) = eleload.loadNerData(dataPath)
    print('初始概率：',initProb)
    print('转移概率：',tranProb)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
