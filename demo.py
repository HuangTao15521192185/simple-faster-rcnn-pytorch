import os
import torch as t
import datetime
import cv2

from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from utils.image_enhance import Image_Enhance
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self, model=''):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        faster_rcnn = FasterRCNNVGG16()
        self.trainer = FasterRCNNTrainer(faster_rcnn).cuda()
        self.trainer.load(model)
        opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model

        self.isneed_enhance = False
        self.imgs_path = '/home/lenovo/4T/Taohuang/VOCdevkit/VOC2007/JPEGImages_bak/'
        self.imgs_vis_path = ''

    def predict(self, imgs=[]):
        imglist = []
        for i in imgs:
            img_path = os.path.join(self.imgs_path, i)
            img = read_image(img_path)
            if self.isneed_enhance:
                img = Image_Enhance().api(img)
            img = t.from_numpy(img)[None]
            imglist.append(img)
        for index, img in enumerate(imglist):
            starttime = datetime.datetime.now()
            _bboxes, _labels, _scores = self.trainer.faster_rcnn.predict(
                img, visualize=True)
            endtime = datetime.datetime.now()
            print('predict time consum=%s' % round(
                (endtime-starttime).microseconds/1000000+(endtime-starttime).seconds, 6))
            if self.imgs_vis_path:
                img_path = os.path.join(self.imgs_vis_path, imgs[index])
                img = read_image(img_path)
                img = t.from_numpy(img)[None]
            ax = vis_bbox(at.tonumpy(img[0]),
                          at.tonumpy(_bboxes[0]),
                          at.tonumpy(_labels[0]).reshape(-1),
                          at.tonumpy(_scores[0]).reshape(-1))
            fig = ax.get_figure()
            fig.savefig("output.png")

    def single_predict(self, img_path=''):
        starttime = datetime.datetime.now()
        img = read_image(img_path)
        img = t.from_numpy(img)[None]
        _bboxes, _labels, _scores = self.trainer.faster_rcnn.predict(
            img, visualize=True)
        endtime = datetime.datetime.now()
        print('predict time consum=%s' % round(
            (endtime-starttime).microseconds/1000000+(endtime-starttime).seconds, 6))
        ax = vis_bbox(at.tonumpy(img[0]),
                      at.tonumpy(_bboxes[0]),
                      at.tonumpy(_labels[0]).reshape(-1),
                      at.tonumpy(_scores[0]).reshape(-1))
        fig = ax.get_figure()
        fig.savefig("output.png")


if __name__ == '__main__':
    model = Model('model_bak/fasterrcnn_03311911_0.8289721270669189_src_enhance')
    model.predict(imgs=['000376.jpg', '000377.jpg','000379.jpg', '000381.jpg', '000382.jpg', '000386.jpg'])
    #model.single_predict('/home/lenovo/4T/Taohuang/simple-faster-rcnn-pytorch/utils/atomization/test.jpg')
