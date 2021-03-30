import os
import torch as t
import datetime

from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
img = read_image('/home/lenovo/4T/Taohuang/VOCdevkit/VOC2007/JPEGImages/000698.jpg')
img = t.from_numpy(img)[None]
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('checkpoints/fasterrcnn_03291629_0.8175883895044492')
opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
starttime = datetime.datetime.now()
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
endtime = datetime.datetime.now()
print('predict time consum=%s' %round((endtime-starttime).microseconds/1000000+(endtime-starttime).seconds,6))
ax=vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
fig = ax.get_figure()
fig.savefig("output.png")
