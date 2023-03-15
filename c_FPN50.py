import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fcn_fpn50
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class gen_fcn_fpn50(nn.Module):
    # change the first layer of the backbone to accept arbitrary number of channels
    def __init__(self, in_channels=3, num_classes=3):
        super(gen_fcn_fpn50, self).__init__()
        self.model = fcn_fpn50(in_channels=in_channels)

        # change the first layer of the backbone to accept arbitrary number of channels
        self.modify_backbone(in_channels)
        # change the number of classes for the RCNN
        self.modify_roi_heads(num_classes)

    def modify_backbone(self, in_channels):
        self.model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def modify_roi_heads(self, num_classes):
        # get the number of input features for the box head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the box predictor with a new one that has the correct number of output classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # add a mask predictor
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features, hidden_layer, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
