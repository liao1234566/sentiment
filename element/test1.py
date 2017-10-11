#-*- coding: utf-8 -*-
'''
Created on 2017年6月21日

@author: Administrator
'''

import codecs

import numpy as np
import pandas as pd
import jieba


# 取两个符号之间的字符串
def get_between(text,start_str,end):
    start=text.find(start_str)
    print(start)
    if start>0:
        start+=len(start_str)
        print(start)
        end=text.find(end)
        print(end)
        if end >=0:
            return text[start:end].strip()

# 获取语料
def get_corpus(input_path,output_path):
    fd=codecs.open(input_path, 'r', 'utf-8')
    lines=fd.readlines()
    fd.close()
    
    output_file_handler=codecs.open(output_path, 'w', 'utf-8')
    for line in lines:
        line=get_between(line, '>', '</')
        new_line=''
        segs=jieba.cut(line,cut_all=False)
        segs=list(segs)
        for word in segs:
            new_line=new_line+word+' '
        output_file_handler.write(new_line.strip() + '\n')
        output_file_handler.flush()
    output_file_handler.close()

if __name__ == '__main__':
    input_path='../../data/element/coae2014.txt'
    output_path='../../data/element/corpus.txt'
    get_corpus(input_path, output_path)
    pass