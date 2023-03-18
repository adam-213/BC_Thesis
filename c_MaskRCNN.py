import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as fpn_maskrcnn50
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNN(nn.Module):
    # change the first layer of the backbone to accept arbitrary number of channels
    def __init__(self, in_channels=3, num_classes=3, mean=None, std=None):
        super(MaskRCNN, self).__init__()

        self.model = fpn_maskrcnn50(in_channels=in_channels,
                                    # pretrained_weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                    num_classes=num_classes,
                                    )

        # change the number of classes for the RCNN
        # self.modify_roi_heads(num_classes)
        self.modify_backbone(in_channels)

        # change the mean and std of the RCNN
        # if this is not done, the model will use the default values which are 3 channel
        # and the tranform will fail on dimension mismatch
        # these values are in the coco dataset created by a_preprocess pipeline
        if mean is None and std is None:
            # these are some "random" values from my testing 2.5k scan dataset assumes full 10 channel
            self.model.transform.image_mean = [0.33320027589797974, 0.3239569664001465, 0.32306283712387085,
                                               0.2266300916671753, 0.2185169756412506, 0.5498420000076294,
                                               0.4774837791919708, 0.4943341016769409, 0.3369252681732178,
                                               0.32673943042755127]
            self.model.transform.image_std = [0.34063756465911865, 0.3318047821521759, 0.33056238293647766,
                                              0.09071272611618042, 0.06958912312984467, 0.317893385887146,
                                              0.12255673110485077, 0.16586355865001678, 0.20912273228168488,
                                              0.3336276412010193]
        else:
            self.model.transform.image_mean = mean
            self.model.transform.image_std = std

    def modify_roi_heads(self, num_classes):
        # get the number of input features for the box and mask heads
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels

        # replace the box and mask heads with new ones that have num_classes which is based on the number of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask,
                                                                dim_reduced=256,
                                                                num_classes=num_classes)

    def modify_backbone(self, in_channels):
        # change the first layer of the backbone to accept arbitrary number of channels
        self.model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def customload(self, path, device, channels=10):
        # Load the pre-trained state dict
        state_dict = torch.load(path, map_location=device)

        if isinstance(channels, (list, np.ndarray, tuple)):
            nc = len(channels)
        else:
            nc = channels
            channels = list(range(channels))

        if nc != 10:
            # Get the weights of the first convolution layer
            conv1_weights = state_dict['backbone.body.conv1.weight']

            # Calculate the mean of the weights along the input channel dimension for the selected channels
            new_conv1_weights = conv1_weights[:, channels, :, :].mean(dim=1, keepdim=True)

            # Update the weights of the first convolution layer in the state dict
            state_dict['backbone.body.conv1.weight'] = new_conv1_weights

        # Load the updated state dict into the current model
        self.model.load_state_dict(state_dict)
