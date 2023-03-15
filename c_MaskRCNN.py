import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as fpn_maskrcnn50
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def normalize(self, image: torch.Tensor) -> torch.Tensor:
    # TODO FIXME: rewrite to accept arbitrary number of channels
    # Turns out this doesn't need to be changed, what needed to be changed is the self.image_mean and self.image_std
    # because if they are None, the code will use the default values which are 3 channel
    if not image.is_floating_point():
        raise TypeError(
            f"Expected input images to be of floating type (in range [0, 1]), "
            f"but found type {image.dtype} instead"
        )
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None]) / std[:, None, None]


class MaskRCNN(nn.Module):
    # change the first layer of the backbone to accept arbitrary number of channels
    def __init__(self, in_channels=3, num_classes=3, mean=None, std=None):
        super(MaskRCNN, self).__init__()

        self.model = fpn_maskrcnn50(in_channels=in_channels,
                                    #pretrained_weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
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

        # replace the box and mask heads with new ones
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask,
                                                                dim_reduced=256,
                                                                num_classes=num_classes)

    def modify_backbone(self, in_channels):
        # change the first layer of the backbone to accept arbitrary number of channels
        self.model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    # def customload(self, path, device, channels=10):
    #     if channels == 10:
    #         self.model.load_state_dict(torch.load(path))
    #         return
    #     elif type(channels) == list or type(channels) == np.array or type(channels) == tuple:
    #         nc = len(channels)
    #         channels = channels
    #     else:
    #         nc = channels
    #         channels = list(range(channels))
    #
    #     # load the state dict
    #     state_dict = torch.load(path)
    #
    #     # TODO finish loading the state dict for arbitrary number of channels
