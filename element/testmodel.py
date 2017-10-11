#-*- coding: utf-8 -*-
'''
Created on 2017年7月3日
模型测试
@author: Administrator
'''
import codecs

import bnermodel


#获取测试集合
def get_testfile(fname,delimiters = [' ', '\n']):
    fd=codecs.open(fname, 'r','utf-8')
    lines=fd.readlines()
    fd.close
    results=[]
    for line in lines:
        line=line.strip()
        if len(line)<=0:continue
        results.append(line)
    return results
    
#对测试结果进行整理规范化
def format_result(lines):
    results=[]
    for line in lines:
        words=line.strip().split()
        if len(words) <=0:continue
        ss=''
        temp_tag=''
        i=0;
        for word in words:
            i=i+1;
            res=word.split('/')
            char=res[0]
            full_tag=res[1]    #处理测试语料时候用
            if full_tag=='O':
                tag=full_tag
            else:
                tag=res[1][0:-2]
            if temp_tag=='':#第一次
                temp_tag=tag
                ss=char
            elif temp_tag==tag and i==len(words):#最后一个标签与前一个相同
                ss=ss+char+'/'+temp_tag
            elif temp_tag!=tag and i==len(words):#最后一个标签与前一个不同
                ss=ss+'/'+temp_tag+'\t'+char+'/'+tag
            elif temp_tag==tag:#如果与前次相同
                ss=ss+char
                temp_tag=tag
            elif temp_tag!=tag or i==len(words):#如果与前次不同
                ss=ss+'/'+temp_tag+'\t'+char
                temp_tag=tag
        results.append(ss)
        ss=''
    return results

#保存处理后的结果
def save_results(path,results):
    output_file_handler=codecs.open(path, 'w', 'utf-8')
    for line in results:
        output_file_handler.write(line.strip()+'\n')
        output_file_handler.flush()
    output_file_handler.close()

#测试语料处理一下，便于计算正确的个数
def test_process(input_path,output_path):
    fd=codecs.open(input_path, 'r', 'utf-8')
    lines=fd.readlines()
    fd.close()
    results=format_result(lines)
    save_results(output_path, results)

#把测试语料处理成预测所需要的格式
def test_input(input_path,output_path):
    fd=codecs.open(input_path, 'r', 'utf-8')
    lines=fd.readlines()
    fd.close()
    output_file_handler=codecs.open(output_path, 'w', 'utf-8')
    for line in lines:
        new_line=''
        words=line.strip().split('\t')
        for word in words:
            res=word.split('/')
            word=res[0]
            new_line=new_line+word
        output_file_handler.write(new_line.strip() + '\n')
        output_file_handler.flush()
    output_file_handler.close()

def test_vocab(input_path):
    fd=codecs.open(input_path, 'r', 'utf-8')
    lines=fd.readlines()
    fd.close()
    p_num=0;f_num=0
    linesvocab=[]
    for line in lines:
        words=line.strip().split()
        if len(words)<1: continue
        vocab={}
        for word in words:
            res=word.split('/')
            char=res[0];tag=res[1]
            if tag=='Product':
                vocab[char]='Product'
                p_num=p_num+1
            elif tag=='Feature':
                vocab[char]='Feature'
                f_num=f_num+1
        linesvocab.append(vocab)
    return p_num,f_num,linesvocab

#计算评测结果
def evaluation(test_compare_path,test_result_path):
    #待比较测试集
    p_num1,f_num1,linesvocab1=test_vocab(test_compare_path)#文档中的总数
    p_num3,f_num3,linesvocab3=test_vocab(test_result_path)#系统找到的的总数
    fd=codecs.open(test_result_path, 'r', 'utf-8')
    lines=fd.readlines()
    fd.close()
    p_num2=0;f_num2=0;#正确识别的个数
    for i in range(len(lines)):
        vocab=linesvocab1[i]
        for word in lines[i].strip().split():
            res=word.split('/')
            char=res[0];tag=res[1]
            if tag=='Product' and char in vocab:
                p_num2=p_num2+1
            elif tag=='Feature' and char in vocab:
                f_num2=f_num2+1
    print('文档中的总数：人名：',p_num1,' 地名：',f_num1)
    print('系统找到的总数：人名：',p_num3,' 地名：',f_num3)
    print('正确识别的个数：人名：',p_num3,' 地名：',f_num3)
    acc_p=p_num2/p_num3;acc_f=f_num2/f_num3;
    recall_p=p_num2/p_num1;recall_f=f_num2/f_num1;
    print(acc_p,recall_p,acc_f,recall_f)
    print('==============F1值============')
    f1_p=(2*acc_p*recall_p)/(acc_p+recall_p)
    f1_f=(2*acc_f*recall_f)/(acc_f+recall_f)
    print('f1_p:',f1_p,"    f1_f:",f1_f)
    return (acc_p,recall_p),(acc_f,recall_f)

if __name__ == '__main__':
    basePath="../../"
    nerModelPath=basePath+'data/element/ele_blstm20.model'
    nerWeightPath=basePath+'data/element/ele_blstm20.weight'
    nerInfoPath=basePath+'data/element/eleInfo_win-7.info'
    nerDataPath=basePath+'data/element/eleData_win-7.data'
    testDataPath=basePath+'data/data/testData.data'
    test_input_path=basePath+'data/element/test_input.txt'
    test_compare_path=basePath+'data/element/ele_test.txt'
    test_result_path=basePath+'data/element/test_result.txt'
    
    # 1.要素预测
    results=bnermodel.testList(get_testfile(test_input_path), nerModelPath,nerWeightPath, nerInfoPath)
    results=format_result(results)
    save_results(test_result_path, results)
    
    #4.要素预测
    evaluation(test_compare_path, test_result_path)
    
    #2.预测结果(输出结果是)
#     input_path=basePath+'data/element/ele_test.txt'
#     output_path=basePath+'data/element/test_compare.txt'
#     test_process(input_path, output_path)

    #3.处理成预测格式
#     input_path=basePath+'data/element/ele_test.txt'
#     output_path=basePath+'data/element/test_input.txt'
#     test_input(input_path, output_path)
    pass