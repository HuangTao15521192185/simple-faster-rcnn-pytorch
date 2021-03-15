# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET

class Dataset_select(object):
    def __init__(self, data_dir='../VOCdevkit/VOC2007/'):
        self.data_dir = data_dir
        self.txt_dir = os.path.join(self.data_dir, 'ImageSets/Main')
        self.jpg_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.xml_dir = os.path.join(self.data_dir, 'Annotations')

    def get_name(self, filename):
        name_list = []
        txt = open(filename)
        for line in txt.readlines():
            if ' 1' in line:
                name = line.split(' ')[0]
                name_list.append(name)
        # print('name=',name_list)
        return name_list

    def get_name_from_filelist(self, filelist):
        name_list = []
        for filename in filelist:
            name_list += self.get_name(filename)
        name_list = set(name_list)
        return name_list

    def delete_other_file(self, filedir, filelist):
        dir_filelist = os.listdir(filedir)
        remove_filelist = sorted(set(dir_filelist)-set(filelist))
        # print('remove_filelist=',remove_filelist)
        for filename in remove_filelist:
            os.remove(os.path.join(filedir, filename))
        return None

    def main(self):
        filelist = ['aeroplane_trainval.txt', 'aeroplane_test.txt', 'bicycle_trainval.txt',
                    'bicycle_test.txt', 'bird_trainval.txt', 'bird_test.txt', 'boat_trainval.txt', 'boat_test.txt']
        name_list = []
        filelist = [os.path.join(self.txt_dir, i) for i in filelist]
        name_list = sorted(self.get_name_from_filelist(filelist))
        print('name_list=', name_list, ',length=', len(name_list))
        xml_list = [i+'.xml' for i in name_list]
        jpg_list = [i+'.jpg' for i in name_list]
        self.delete_other_file(filedir=self.xml_dir, filelist=xml_list)
        # self.delete_other_file(filedir=self.jpg_dir, filelist=jpg_list)

    def delete_file_from_xml(self, xml_file_list):
        delete_list=[]
        for xml_file in xml_file_list:
            anno = ET.parse(
                os.path.join(self.data_dir, 'Annotations', xml_file))
            for obj in anno.findall('object'):
                name = obj.find('name').text.lower().strip()
                if name not in ('aeroplane', 'bicycle', 'bird', 'boat'):
                    delete_list.append(xml_file)
                    break
        print('wrong delete_list=', delete_list)        
        print('length=',len(delete_list))
        for filename in delete_list:
            os.remove(os.path.join(self.data_dir, 'Annotations', filename))
        return delete_list


if __name__ == "__main__":
    dataset_select = Dataset_select()
    dataset_select.main()
