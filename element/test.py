#-*- coding: utf-8 -*-
'''
Created on 2017年6月10日

@author: Administrator
'''
import codecs

def get_text(input_path,output_path):
    fd=codecs.open(input_path, 'r', 'utf-8')
    lines=fd.readlines()
    fd.close()
    
    output_file=codecs.open(output_path,'w','utf-8')
    
    for line in lines:
        new_line=''
        words=line.strip().split('\t')
        for word in words:
            res=word.split()
            if res[0]=='O':continue
            new_line=new_line+res[0]+' '
        output_file.write(new_line.strip()+'\n')
        output_file.flush()
    output_file.close

if __name__ == '__main__':
    input_path='../../data/element/pos_neg.txt'
    output_path='../../data/element/char.txt'
    get_text(input_path, output_path)
    pass