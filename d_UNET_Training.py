import time

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch

from b_DataLoader_RCNN import createDataLoader
from c_UNET import UNet_InstanceSegmentation

dataset, dataloader = createDataLoader()
model = UNet_InstanceSegmentation(3, 32)

scaler = GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.cuda()
lossfn = torch.nn.CrossEntropyLoss()

for i, batch in enumerate(tqdm(dataloader)):
    with autocast():
        images, masks = batch
        images = images.cuda()
        masks = masks.cuda()

        # forward pass
        outputs = model(images)
        print(outputs)
        print(type(outputs))


        # loss
        loss = lossfn(outputs["out"], masks)

    # backward pass
    scaler.scale(loss).backward()

    # update weights
    scaler.step(optimizer)
    scaler.update()
