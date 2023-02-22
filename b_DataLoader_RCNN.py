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
import torch
import numpy as np
from pycocotools import mask as coco_mask


# Custom loader is needed to load RGB-A images - (4 channels) which in my case are RGB-D images
class CustomCocoDetection(CocoDetection):

    def _load_image(self, id: int) -> torch.Tensor:
        # overrride the _load_image method to load the images from the npz files in fp16
        path = self.coco.loadImgs(id)[0]["file_name"]
        npz_file = np.load(os.path.join(self.root, path))
        img_arrays = [npz_file[key] for key in npz_file.keys()]
        # dstack with numpy because pytorch refuses to stack different number of channels reasons
        image = np.dstack(img_arrays).astype(np.float16)
        # IF FULL
        # R,G,B, D, NX,NY,NZ ,I
        return torch.from_numpy(image).type(torch.float16)


def prepare_targets(targets) -> list:
    prepared_targets = []
    for target in targets:
        prepared_target = {}

        # Masks
        inst_masks = []
        inst_labels = []
        for inst in target:
            inst_mask = torch.from_numpy(coco_mask.decode(eval(inst['segmentation']))).bool()
            inst_masks.append(inst_mask)
            inst_labels.append(inst['id'])

            # Pad masks to all be the same size, needed for torch to stack them
        inst_masks = torch.nn.utils.rnn.pad_sequence(inst_masks, batch_first=True, padding_value=0)

        # Transformation Matrices
        inst_transforms = [torch.tensor(inst['transform']).view(4, 4) for inst in target]
        inst_transforms = torch.stack(inst_transforms, dim=0)

        # Bounding Boxes
        bbox = [torch.tensor([inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
                              inst['bbox'][1] + inst['bbox'][3]]).view(1, 4) for inst in target]
        bbox = torch.cat(bbox, dim=0)

        # Covert to correct data types
        inst_masks = inst_masks.type(torch.bool)  # binary masks
        inst_transforms = inst_transforms.type(torch.float32)  # tiny, can stay in float32
        bbox = bbox.type(torch.float16)  # fp32 is not needed for bbox

        # Store in dictionary
        prepared_target['masks'] = inst_masks
        prepared_target['transforms'] = inst_transforms
        prepared_target['boxes'] = bbox
        prepared_target['area'] = torch.tensor([inst['area'] for inst in target])
        # labels need to be int64, such overkill
        prepared_target['labels'] = torch.tensor(inst_labels, dtype=torch.int64)

        # Don't know if necessary
        # prepared_target['image_id'] = torch.tensor([target[0]['image_id']])

        # Not defined - not needed
        # prepared_target['iscrowd'] = torch.zeros((bbox.shape[0],), dtype=torch.int8)

        prepared_targets.append(prepared_target)

    # List of dictionaries - one dictionary per image
    return prepared_targets


def collate_fn(batch):
    """Custom collate function to prepare data for the model"""
    images, targets = zip(*batch)

    # Stack images on first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    # TODO Figure this out but for now choose 3 channels
    batched_images = batched_images[:, [0, 1, 3], :, :]
    # For some reason this needs to be under the 3 channel selection
    batched_images.requires_grad_(True)  # RCNN needs gradients on input images

    # Prepare targets
    prepared_targets = prepare_targets(targets)

    # Return batch as a tuple
    return batched_images, prepared_targets


def createDataLoader():
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO')
    ano_path = coco_path.joinpath('annotations')
    # Load the COCO dataset
    dataset = CustomCocoDetection(root=str(coco_path), annFile=str(ano_path.joinpath('coco.json')))

    # Create a PyTorch dataloader
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2, collate_fn=collate_fn)

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
