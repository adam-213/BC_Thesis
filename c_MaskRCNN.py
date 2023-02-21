import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as fpn_maskrcnn50
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights


class MaskRCNN(nn.Module):
    # change the first layer of the backbone to accept arbitrary number of channels
    def __init__(self, in_channels=3, num_classes=3):
        super(MaskRCNN, self).__init__()
        self.model = fpn_maskrcnn50(in_channels=in_channels,
                                    pretrained_weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

        # change the number of classes for the RCNN
        self.modify_roi_heads(num_classes)

    def modify_roi_heads(self, num_classes):
        # get the number of input features for the box and mask heads
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels

        # replace the box and mask heads with new ones
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask,
                                                                dim_reduced=256,
                                                                num_classes=num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
