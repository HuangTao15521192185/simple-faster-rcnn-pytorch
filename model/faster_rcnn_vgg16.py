from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)
    features = list(model.features)[:30]
    if opt.use_multi_conv:
        features.append(Inception(256, 128, (128, 256), 64, (24, 64)))
        features.append(nn.ReLU(inplace=True))
        features.append(ResNetA(512,128,(128,256),512,128,(128,256),128,128,(128,256),512))
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.p3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_c, c3, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(in_c, c4[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4[0], c4[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4[1], c4[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        normal_init(self.p1[0], 0, 0.01)
        normal_init(self.p2[0], 0, 0.01)
        normal_init(self.p2[2], 0, 0.01)
        normal_init(self.p3[1], 0, 0.01)
        normal_init(self.p4[0], 0, 0.01)
        normal_init(self.p4[2], 0, 0.01)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return t.cat((p1, p2, p3, p4), dim=1)


class ResNetA(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4, c5, c6, c7, c8, c9):
        super(ResNetA, self).__init__()
        # ResNetA
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[1], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(c1+c2[1], c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # ReducionA
        self.p4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(c3, c5[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5[0], c5[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5[1], c5[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.p6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c3, c6, kernel_size=1)
        )
        # ResNetB
        self.p7 = nn.Sequential(
            nn.Conv2d(c4+c5[1]+c6, c7, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.p8 = nn.Sequential(
            nn.Conv2d(c4+c5[1]+c6, c8[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c8[0], c8[1], kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c8[1], c8[1], kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(inplace=True)
        )
        self.p9 = nn.Sequential(
            nn.Conv2d(c7+c8[1], c9, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)
        normal_init(self.p1[0], 0, 0.01)
        normal_init(self.p2[0], 0, 0.01)
        normal_init(self.p2[2], 0, 0.01)
        normal_init(self.p2[4], 0, 0.01)
        normal_init(self.p3[0], 0, 0.01)
        normal_init(self.p4[0], 0, 0.01)
        normal_init(self.p5[0], 0, 0.01)
        normal_init(self.p5[2], 0, 0.01)
        normal_init(self.p5[4], 0, 0.01)
        normal_init(self.p6[1], 0, 0.01)
        normal_init(self.p7[0], 0, 0.01)
        normal_init(self.p8[0], 0, 0.01)
        normal_init(self.p8[2], 0, 0.01)
        normal_init(self.p8[4], 0, 0.01)
        normal_init(self.p9[0], 0, 0.01)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(t.cat((p1, p2), dim=1))
        resnetA = self.relu(x+p3)
        p4 = self.p4(resnetA)
        p5 = self.p5(resnetA)
        p6 = self.p6(resnetA)
        reducionA = self.relu(t.cat((p4, p5, p6), dim=1))
        p7 = self.p7(reducionA)
        p8 = self.p8(reducionA)
        p9 = self.p9(t.cat((p7, p8), dim=1))
        resnetB = self.relu(reducionA+p9)
        return resnetB


class ResNetB(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4, c5, c6, c7):
        super(ResNetB, self).__init__()
        # ReductionB
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_c, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(in_c, c4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c4, c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # ResNetC
        self.p5 = nn.Sequential(
            nn.Conv2d(c1+c2[1]+c3+c4, c5, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.p6 = nn.Sequential(
            nn.Conv2d(c1+c2[1]+c3+c4, c6[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c6[0], c6[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c6[1], c6[1], kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.p7 = nn.Sequential(
            nn.Conv2d(c5+c6[1], c7, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        normal_init(self.p1[0], 0, 0.01)
        normal_init(self.p2[0], 0, 0.01)
        normal_init(self.p2[2], 0, 0.01)
        normal_init(self.p3[0], 0, 0.01)
        normal_init(self.p3[2], 0, 0.01)
        normal_init(self.p4[0], 0, 0.01)
        normal_init(self.p4[3], 0, 0.01)
        normal_init(self.p5[0], 0, 0.01)
        normal_init(self.p6[0], 0, 0.01)
        normal_init(self.p6[2], 0, 0.01)
        normal_init(self.p6[4], 0, 0.01)
        normal_init(self.p7[0], 0, 0.01)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        reductionB = self.relu(t.cat((p1, p2, p3, p4), dim=1))
        p5 = self.p5(reductionB)
        p6 = self.p6(reductionB)
        p7 = self.p7(t.cat((p5, p6), dim=1))
        resnetC = self.relu(reductionB+p7)
        return self.avgpool(resnetC)


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=opt.label_number,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):

        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        #expand_field = ResNetB(512,128, (128, 256), 64, 64, 128, (128,256),512)

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
            mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
