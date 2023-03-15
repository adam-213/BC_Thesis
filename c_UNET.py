import torch.nn as nn
import torchvision.models as models


class UNet_InstanceSegmentation(nn.Module):
    def __init__(self, num_classes=2, num_instances=2):
        super(UNet_InstanceSegmentation, self).__init__()

        # Load the pre-trained UNet model
        self.base_model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)

        self.base_model.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the last layer to output a binary mask for each object instance
        # self.base_model.classifier = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, num_classes + num_instances, kernel_size=1)
        # )

    def forward(self, x):
        return self.base_model(x)
