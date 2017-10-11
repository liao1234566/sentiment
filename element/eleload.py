#-*- coding: utf-8 -*-
'''
Created on 2017年6月20日

@author: Administrator
'''
import json

import h5py


# 载入情感要素训练数据字典和概率
def loadNerInfo(path):
    print('load cws info from %s'%path)
    fd = open(path, 'r',-1,'utf-8')
    line = fd.readline()
    print(line)
    j = json.loads(line.strip())
    initProb, tranProb = j[0], j[1]
    lines = fd.readlines()
    fd.close()
    vocab = {}
    indexVocab = [0 for i in range(len(lines))]
    for line in lines:
        rst = line.strip().split('\t')
        if len(rst) < 2: continue
        char, index = rst[0], int(rst[1])
        # print(rst[0])
        vocab[char] = index
        indexVocab[index] = char
    return (initProb, tranProb), (vocab, indexVocab)

# 载入情感要素训练输入样本
def loadNerData(path):
    print('load cws data from %s'%path)
    fd = h5py.File(path,'r')
    X = fd['X'][:]
    y = fd['y'][:]
    fd.close()
    return (X, y)
if __name__ == '__main__':
    pass