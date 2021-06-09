# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
import re


class Dataset_tool(object):
    def __init__(self):
        self.path = '/home/lenovo/4T/Taohuang/simple-faster-rcnn-pytorch/utils/atomization/'

    def image_resize(self):
        Start_path = self.path
        iphone5_width = 600  # 图片最大宽度
        iphone5_depth = 1000  # 图片最大高度

        list = os.listdir(Start_path)
        # print list
        count = 0
        for pic in list:
            path = Start_path+pic
            im = Image.open(path)
            w, h = im.size
            # print w,h
            # iphone 5的分辨率为1136*640，如果图片分辨率超过这个值，进行图片的等比例压缩

            if w > iphone5_width:
                h_new = iphone5_width*h/w
                w_new = iphone5_width
                count = count+1
                out = im.resize((int(w_new), int(h_new)), Image.ANTIALIAS)
                new_pic = re.sub(pic[:-4], pic[:-4], pic)
                # print new_pic
                new_path = Start_path+new_pic
                out.save(new_path)
            if h > iphone5_depth:
                w = iphone5_depth*w/h
                h = iphone5_depth
                count = count+1
                out = im.resize((int(w_new), int(h_new)), Image.ANTIALIAS)
                new_pic = re.sub(pic[:-4], pic[:-4], pic)
                # print new_pic
                new_path = Start_path+new_pic
                out.save(new_path)

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0  # 图片编号从多少开始，不要跟VOC原本的编号重复了
        n = 6
        for item in filelist:
            n = 6 - len(str(i))
            src = os.path.join(os.path.abspath(self.path), item)
            dst = os.path.join(os.path.abspath(self.path),
                               str(0)*n + str(i) + '.jpg')
            try:
                os.rename(src, dst)
                i = i + 1
            except:
                continue

    def replace_img(self, src_path, dst_path, txt_path):
        f = open(txt_path, 'r')
        imglist = [i.replace('\n', '') for i in f]
        f.close()
        for img in imglist:
            cmd = "cp %s %s" % (os.path.join(
                src_path, img+'.jpg'), os.path.join(dst_path, img+'.jpg'))
            print('cmd=', cmd)
            os.system(cmd)
            
    def count_dataset(self,annotizations_path):
        filelist = os.listdir(annotizations_path)
        count={'animal':0,'plant':0,'rock':0}
        for xml in filelist:
            f = open(os.path.join(annotizations_path,xml),'r')
            lines = f.readlines()
            live={'animal':False,'plant':False,'rock':False}
            for line in lines:
                if '/name' in line:
                    for key in count.keys():
                        if key in line:
                            if live[key] is False:
                                count[key]+=1
                                live[key]=True
                            break
            f.close()
        print(count)
    def divide_dataset(self,txt_index,annotizations,jpeg):
        txt_list=['trainval','test']
        for txt in txt_list:
            os.system('mkdir -p %s/%s' %(annotizations,txt))
            os.system('mkdir -p %s/%s' %(jpeg,txt))
            f = open(os.path.join(txt_index,txt+'.txt'),'r')
            lines = f.readlines()
            for line in lines:      
                xml = os.path.join(annotizations,line.replace('\n','')+'.xml')
                jpg = os.path.join(jpeg,line.replace('\n','')+'.jpg')
                cmd1 = 'cp  %s %s/%s/' %(xml,annotizations,txt)
                cmd2 = 'cp  %s %s/%s/' %(jpg,jpeg,txt)
                print('cmd1=',cmd1)
                print('cmd2=',cmd2)
                os.system(cmd1)
                os.system(cmd2)




def random_filename(filepath='.'):
    filelist = os.listdir(filepath)
    for filename in filelist:
        newname = '0'+filename
        os.rename(os.path.join(filepath, filename),
                  os.path.join(filepath, newname))


if __name__ == '__main__':
    # random_filename('./dataset')
    dataset_tool = Dataset_tool()
    dataset_tool.divide_dataset(txt_index='/home/lenovo/4T/Taohuang/underwater-object-detection-mmdetection/data/Main',
    annotizations='/home/lenovo/4T/Taohuang/underwater-object-detection-mmdetection/data/train/box',
    jpeg='/home/lenovo/4T/Taohuang/underwater-object-detection-mmdetection/data/train/image')
