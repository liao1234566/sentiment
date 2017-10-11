#-*- coding: utf-8 -*-
'''
Created on 2017年6月21日
    极性预测
@author: Administrator
'''

import codecs

from gensim.models.word2vec import Word2Vec
import jieba
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier


# 获取训练数据
def get_train(fname,stop_fname,model_path):
    col_names=['lable','data']
    dataset=pd.read_csv(fname,header=None,names=col_names,delimiter='\t')
    stoplist=codecs.open(stop_fname, 'r', 'utf-8').readlines()#读取停用词
    stoplist=set( w.strip() for w in stoplist)# 放set中去重
    model=Word2Vec.load(model_path)
    #数据和标签划分
    X=dataset['data']
    y=dataset['lable']
    
    X_train=[]
    for i in X:
        line2vec=np.zeros(model.vector_size)
        segs=jieba.cut(i,cut_all=False)
        segs=[ word for word in list(segs) if word not in stoplist]
        count=0
        for word in segs:
            if word in model:
                line2vec=line2vec+model[word]
                count+=1
        if count!=0:
            line2vec=line2vec*(1.0/count)#取平均值
            X_train.append(list(line2vec))
    return X_train,y

# svm 训练
def svm_train(X,y,model_path):
    model=SVC()
    model.fit(X,y)
    expected=y
    predicted=model.predict(X)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    joblib.dump(model,model_path)

#逻辑回归
def lg(X,y,model_path):
    model=LogisticRegression()
    model.fit(X, y)
    print(model)
    expected=y
    predicted=model.predict(X)
    print(metrics.classification_report(expected,predicted))
    print(metrics.confusion_matrix(expected, predicted))
    joblib.dump(model,model_path)

#朴素贝叶斯
def bayes(X,y,model_path):
    model=GaussianNB()
    model.fit(X,y)
    expected=y
    predicted=model.predict(X)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    joblib.dump(model,model_path)

# K 近邻
def knn(X,y,model_path):
    model=KNeighborsClassifier()
    model.fit(X,y)
    print(model)
    #预测
    expected=y
    predicted=model.predict(X)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    joblib.dump(model,model_path)
    
#决策树
def dtree(X,y,model_path):
    model=DecisionTreeClassifier()
    model.fit(X,y)
    expected=y
    predicted=model.predict(X)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    joblib.dump(model,model_path)

#随机森林
def forest(X,y,model_path):
    model=RandomForestClassifier()
    model.fit(X,y)
    expected=y
    predicted=model.predict(X)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    joblib.dump(model,model_path)

# 句子转为向量（去除停用词）
def word_vec(sents,stop_path,vecmodel_path):
    stoplist=codecs.open(stop_path, 'r', 'utf-8').readlines()#读取停用词
    stoplist=set( w.strip() for w in stoplist)# 放set中去重
    vecmodel=Word2Vec.load(vecmodel_path)
    X=[]
    for sent in sents:
        segs=jieba.cut(sent,cut_all=False)
        segs=[ word for word in list(segs) if word not in stoplist]
        count=0
        sent2vec=np.zeros(vecmodel.vector_size)
        for word in segs:
            if word in vecmodel:
                sent2vec=sent2vec+vecmodel[word]
                count+=1
        if count!=0:
            sent2vec=sent2vec*(1.0/count)
            X.append(list(sent2vec))
    return X

# 测试
def test(sents,model_path,stop_path,vecmodel_path):
    model=joblib.load(model_path)
    X=word_vec(sents, stop_path, vecmodel_path)
    predicts=[]
    for x in X:
        predict=model.predict(x)
        predicts.append(predict)
    return predicts
    
if __name__ == '__main__':
    base_path='../../'
#     base_path="/home/liguangcai/zutnlp-py/zutnlp/"
    
    fname=base_path+'data/element/polarity.data'
    stop_path=base_path+'data/element/stopwords.txt'
    vecmodel_path=base_path+'data/element/w2vec.model'
    X,y=get_train(fname,stop_path,vecmodel_path)
    print('-------svm---------')
    svm_model_path=base_path+'data/element/svm.model'
#     svm_train(X, y,svm_model_path)
    sent=['好好喝 谢谢蒙牛琪琪曼曼苗苗今天很成功','坚决抵制蒙牛牛奶,坚决抵制！宁愿去喝妇炎洁。']
    predicts=test(sent, svm_model_path, stop_path, vecmodel_path)
    print(predicts)
    
    print('-------逻辑回归---------')
    lg_model_path=base_path+'data/element/lg.model'
#     lg(X, y, lg_model_path)
    sent=['好好喝 谢谢蒙牛琪琪曼曼苗苗今天很成功','坚决抵制蒙牛牛奶,坚决抵制！宁愿去喝妇炎洁。']
    predicts=test(sent, lg_model_path, stop_path, vecmodel_path)
    print(predicts)
    
    print('-------朴素贝叶斯---------')
    bayes_model_path=base_path+'data/element/bayes.model'
#     bayes(X, y, bayes_model_path)
    sent=['好好喝 谢谢蒙牛琪琪曼曼苗苗今天很成功','坚决抵制蒙牛牛奶,坚决抵制！宁愿去喝妇炎洁。']
    predicts=test(sent, bayes_model_path, stop_path, vecmodel_path)
    print(predicts)
    
    print('-------K 近邻---------')
    knn_model_path=base_path+'data/element/knn.model'
#     knn(X,y,knn_model_path)
    sent=['好好喝 谢谢蒙牛琪琪曼曼苗苗今天很成功','坚决抵制蒙牛牛奶,坚决抵制！宁愿去喝妇炎洁。']
    predicts=test(sent, knn_model_path, stop_path, vecmodel_path)
    print(predicts)
    
    print('-------决策树---------')
    dtree_model_path=base_path+'data/element/dtree.model'
#     dtree(X,y,dtree_model_path)
    sent=['好好喝 谢谢蒙牛琪琪曼曼苗苗今天很成功','坚决抵制蒙牛牛奶,坚决抵制！宁愿去喝妇炎洁。']
    predicts=test(sent, dtree_model_path, stop_path, vecmodel_path)
    print(predicts)
    
    forest_model_path=base_path+'data/element/forest.model'
#     forest(X, y, forest_model_path)
    sent=['好好喝 谢谢蒙牛琪琪曼曼苗苗今天很成功','坚决抵制蒙牛牛奶,坚决抵制！宁愿去喝妇炎洁。']
    predicts=test(sent, dtree_model_path, stop_path, vecmodel_path)
    print(predicts)
    pass



