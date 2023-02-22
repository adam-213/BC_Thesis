import os
import pathlib
import time

import torch
import torchvision
import torchvision.transforms
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection


# Custom loader is needed to load RGB-A images - (4 channels) which in my case are RGB-D images
class CustomCocoDetection(CocoDetection):
    # overrride the _load_image method
    # change this when trying to load say npz files with many channels as input
    # rgba png
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGBA")



def collate_fn(batch):
    """This function needs to output a tuple of tensors (images, targets)
    images is a tensor of shape (batch_size, channels, H, W)
    targets is a tensor of shape (batch_size, max_num_instances, H,W)
    max_num_instances is the maximum number of instances in the batch
    this is needed to pad the rest of the instances with zeros
    realistically this dimension is the whole decoded binary mask for the image and instance pair
    where instance = a single part
    """
    # Pad the rest of the instances with zeros
    # Question is how to pad the rest of the instances with zeros
    # since this isn't a tensor yet
    # do i even care ? Convert everything to tensor and then pad with zeros ?
    image, targets = zip(*batch)
    # convert the images to a tensor
    image = [torchvision.transforms.ToTensor()(img).type(torch.uint8) for img in image]

    # turn the list of images into a tensor
    image_tensor = torch.stack(image, dim=0)
    del image
    # check the shape of the image tensor
    # print(image.shape)
    assert image_tensor.shape[1] == 4, "The image tensor should have 4 channels (RGBA)"
    assert image_tensor.shape[0] == len(
        batch), "The image tensor should have the same number of images as the batch size"
    # un-rle the targets
    # Convert RLE-encoded binary masks to tensors and store their shapes
    masks = []
    mask_shapes = []
    transforms = []
    bboxs = []

    for target in targets:
        inst_masks = [torch.from_numpy(coco_mask.decode(eval(inst['segmentation']))).bool().unsqueeze(0) for inst in
                      target]
        # conver the inst_masks to a sparse tensor
        inst_masks = torch.cat(inst_masks, dim=0)

        inst_transforms = [torch.tensor(inst['transform']).view(4, 4).unsqueeze(0) for inst in target]

        mask_shapes.append(len(inst_masks))

        bbox = [torch.tensor(inst['bbox']).view(1, 4) for inst in target]

        # convert to sparse tensor
        masks.append(inst_masks)
        transforms.append(torch.cat(inst_transforms, dim=0))
        bboxs.append(torch.cat(bbox, dim=0))

    # Pad the masks to the maximum number of instances in the batch
    masks_tensor = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    # sparse_tensor = masks_tensor.to_sparse(3)
    # del masks, masks_tensor

    # Pad the transforms to the maximum number of instances in the batch
    transforms_tensor = torch.nn.utils.rnn.pad_sequence(transforms, batch_first=True, padding_value=0)

    # Pad the bboxs to the maximum number of instances in the batch
    # bboxs_tensor = torch.nn.utils.rnn.pad_sequence(bboxs, batch_first=True, padding_value=0)

    # scale the images to the range [0,1] and convert to float16
    image_tensor = image_tensor.type(torch.float16) / 255.0

    return image_tensor, masks_tensor


import torch
import numpy as np
from pycocotools import mask as coco_mask


def collate_fn2(batch):
    images, targets = zip(*batch)
    # convert the images to a tensor
    images = [torchvision.transforms.ToTensor()(img).type(torch.uint8) for img in images]
    # turn the list of images into a tensor
    image_tensor = torch.stack(images, dim=0)

    image_tensor = image_tensor.type(torch.float16) / 255.0
    # check the shape of the image tensor
    assert image_tensor.shape[1] == 4, "The image tensor should have 4 channels (RGBA)"
    assert image_tensor.shape[0] == len(
        batch), "The image tensor should have the same number of images as the batch size"

    # todo delete this
    image_tensor = image_tensor[:, [0, 1, 3], :, :]

    # stack 2 of the image_tensors on the channel dimension
    # image_tensor = torch.cat([image_tensor, image_tensor], dim=1)
    # removing the either blue or red channel depending on the Rgb vs bgr

    # Prepare targets
    prepared_targets = []
    for target in targets:
        # Initialize dictionary to store this target's information
        prepared_target = {}

        # Convert RLE-encoded binary masks to tensors and store their shapes
        inst_masks = []
        mask_shapes = []
        for inst in target:
            inst_mask = torch.from_numpy(coco_mask.decode(eval(inst['segmentation']))).bool()
            inst_masks.append(inst_mask)
            mask_shapes.append(inst_mask.shape)
        inst_masks = torch.nn.utils.rnn.pad_sequence(inst_masks, batch_first=True, padding_value=0)
        mask_shapes = torch.tensor(mask_shapes)

        # Convert transforms to tensor
        inst_transforms = [torch.tensor(inst['transform']).view(4, 4) for inst in target]
        inst_transforms = torch.stack(inst_transforms, dim=0)

        # Convert bounding boxes to tensor
        bbox = [torch.tensor([inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
                              inst['bbox'][1] + inst['bbox'][3]]).view(1, 4) for inst in target]
        bbox = torch.cat(bbox, dim=0)

        # convert everything to float16
        inst_masks = inst_masks.type(torch.bool)
        # mask_shapes = mask_shapes.type(torch.float16)
        inst_transforms = inst_transforms.type(torch.float16)
        bbox = bbox.type(torch.float16)

        # Set fields in prepared target dictionary
        prepared_target['masks'] = inst_masks
        # prepared_target['mask_shapes'] = mask_shapes
        prepared_target['boxes'] = bbox
        prepared_target['labels'] = torch.ones((bbox.shape[0],), dtype=torch.int64)  # COCO only has 1 class
        prepared_target['image_id'] = torch.tensor([target[0]['image_id']])
        prepared_target['area'] = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])

        prepared_targets.append(prepared_target)

    # print("Mask shapes: ", inst_masks.shape)
    # print("Image shape: ", image_tensor.shape)
    # torch.cuda.reset_max_memory_allocated()
    # print("\nb4 to cudab4 chache",torch.cuda.max_memory_allocated())
    # torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated()
    # print("\nb4 to cuda", torch.cuda.max_memory_allocated())

    # Return batch as dictionary
    image_tensor.requires_grad_(True)
    return image_tensor, prepared_targets


def createDataLoader():
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO')
    ano_path = coco_path.joinpath('annotations')
    # Load the COCO dataset
    dataset = CustomCocoDetection(root=str(coco_path), annFile=str(ano_path.joinpath('coco.json')))

    # Create a PyTorch dataloader
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=1, collate_fn=collate_fn2)

    return dataset, dataloader


if __name__ == '__main__':
    # set seeds
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # import random
    #
    # random.seed(0)
    dataset, dataloader = createDataLoader()

    # check the first batch
    from tqdm import tqdm

    for i, batch in enumerate(tqdm(dataloader)):
        images = batch[0].cuda()
        masks = batch[1].cuda()
        transforms = batch[2].cuda()
        bboxs = batch[3]
        del batch
