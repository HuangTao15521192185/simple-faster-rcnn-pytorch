# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random  
  
trainval_percent = 0.7  #trainval占比例多少
train_percent = 0.3  #train数据集占trainval比例多少
xmlfilepath = '../VOCdevkit/VOC2007/Annotations'  
txtsavepath = '../VOCdevkit/VOC2007/ImageSets/Main'  
# xmlfilepath = '../underwater-object-detection-mmdetection/data/train/box'  
# txtsavepath = '../underwater-object-detection-mmdetection/data/Main'
total_xml = os.listdir(xmlfilepath)  
  
num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
  
ftrainval = open('../VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')  
ftest = open('../VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')  
ftrain = open('../VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')  
fval = open('../VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')  

# ftrainval = open('../underwater-object-detection-mmdetection/data/Main/trainval.txt', 'w')  
# ftest = open('../underwater-object-detection-mmdetection/data/Main/test.txt', 'w')  
# ftrain = open('../underwater-object-detection-mmdetection/data/Main/train.txt', 'w')  
# fval = open('../underwater-object-detection-mmdetection/data/Main/val.txt', 'w') 
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()  