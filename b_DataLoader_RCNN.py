import os
import pathlib

import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from matplotlib import patches


# Custom loader is needed to load RGB-A images - (4 channels) which in my case are RGB-D images
class CustomCocoDetection(CocoDetection):

    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile, transforms)

        image_stats = self.coco.dataset.get("image_stats", {})
        self.mean = image_stats.get("mean", [0.485, 0.456, 0.406])
        self.std = image_stats.get("std", [0.229, 0.224, 0.225])

    def _load_image(self, id: int) -> torch.Tensor:
        # overrride the _load_image method to load the images from the npz files in fp16
        # print("Loading image: ", id, "")
        path = self.coco.loadImgs(id)[0]["file_name"]
        # print("Path: ", path, " ")
        npz_file = np.load(os.path.join(self.root, path))
        img_arrays = [npz_file[key] for key in npz_file.keys()]
        # dstack with numpy because pytorch refuses to stack different number of channels reasons
        image = np.dstack(img_arrays).astype(np.float32)
        # IF FULL
        # R,G,B, X,Y,Z, NX,NY,NZ ,I
        return torch.from_numpy(image).type(torch.float32)

    def _load_target(self, id: int):
        # override the _load_target becaiuse the original one is doing some weird stuff
        # no idea why it is doing that
        x = self.coco.imgToAnns[id]
        return x


def prepare_targets(targets) -> list:
    prepared_targets = []
    for target in targets:
        prepared_target = {}

        # Masks
        inst_masks = []
        inst_labels = []
        for inst in target:
            segmentation = eval(inst['segmentation'])
            # print('Segmentation:', segmentation)

            decoded_mask = coco_mask.decode(segmentation)
            inst_mask = torch.from_numpy(decoded_mask).bool()
            # print('Decoded mask:', decoded_mask)
            # print('Inst mask:', inst_mask)

            inst_masks.append(inst_mask)
            inst_labels.append(inst['id'])

            # debug
            # plot inst mask and bbox
            # fig, ax = plt.subplots(1)
            # ax.imshow(inst_mask)
            # bbox = [inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
            #         inst['bbox'][1] + inst['bbox'][3]]
            # ax.imshow(inst_mask)
            # ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            #                                       linewidth=1, edgecolor='r', facecolor='none'))
            # plt.show()

        # print('Inst masks:', inst_masks)
        # print('Inst labels:', inst_labels)

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
        bbox = bbox.type(torch.float32)  # fp32 is not needed for bbox

        # Store in dictionary
        prepared_target['masks'] = inst_masks.float()
        prepared_target['transforms'] = inst_transforms.float()
        prepared_target['boxes'] = bbox.float()
        prepared_target['area'] = torch.tensor([inst['area'] for inst in target]).float()
        # labels need to be int64, such overkill
        prepared_target['labels'] = torch.tensor(inst_labels, dtype=torch.int64)

        # Don't know if necessary
        # prepared_target['image_id'] = torch.tensor([target[0]['image_id']])

        # Not defined - not needed
        # prepared_target['iscrowd'] = torch.zeros((bbox.shape[0],), dtype=torch.int8)

        prepared_targets.append(prepared_target)

        # # plot the masks and bounding boxes using patches
        # fig, ax = plt.subplots(1)
        # #ax.imshow(image)
        # for i in range(len(inst_masks)):
        #     mask = inst_masks[i]
        #     box = bbox[i]
        #     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
        #                              facecolor='none')
        #     ax.add_patch(rect)
        #     ax.imshow(mask, alpha=0.5)
        #     plt.show()

    # List of dictionaries - one dictionary per image
    return prepared_targets


def prepare_targets_heuristic(targets, k=5) -> list:
    prepared_targets = []
    for target in targets:
        prepared_target = {}

        # Masks
        inst_masks = []
        inst_labels = []
        inst_areas = []
        backbin_masks = []
        backbin_labels = []
        backbin_areas = []
        for inst in target:
            if inst['category_id'] not in [0, 1]:  # exclude background and bin
                segmentation = eval(inst['segmentation'])
                decoded_mask = coco_mask.decode(segmentation)
                inst_mask = torch.from_numpy(decoded_mask).bool()
                inst_masks.append(inst_mask)
                inst_labels.append(inst['id'])
                inst_areas.append(inst['area'])
            else:
                segmentation = eval(inst['segmentation'])
                decoded_mask = coco_mask.decode(segmentation)
                backbin = torch.from_numpy(decoded_mask).bool()
                backbin_masks.append(backbin)
                backbin_labels.append(inst['id'])
                backbin_areas.append(inst['area'])
                # todo thik about adding the background and bin to the instance masks
                # somehow marking bin as empty so we can predit stop

        if len(inst_masks) == 0:
            # set everything to nothing if no instance masks with th correct dimensions
            inst_masks = torch.zeros((1, 772, 1032), dtype=torch.bool)
            inst_labels = torch.zeros((999,), dtype=torch.int64)
            inst_areas = torch.zeros((1,), dtype=torch.float32)
            inst_transforms = torch.zeros((1, 4, 4), dtype=torch.float32)
            # use full image as bbox
            bbox = torch.tensor([[0, 0, 10, 10]]).float()
        else:

            # Choose the top k masks by area
            idx = torch.argsort(torch.tensor(inst_areas), descending=True)[:k]
            inst_masks = [inst_masks[i] for i in idx]
            inst_masks = torch.nn.utils.rnn.pad_sequence(inst_masks, batch_first=True, padding_value=0)
            inst_transforms = torch.stack(
                [torch.tensor(inst['transform']).view(4, 4) for i, inst in enumerate(target) if i in idx], dim=0)

            bbox = [torch.tensor([inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
                                  inst['bbox'][1] + inst['bbox'][3]]).view(1, 4) for i, inst in enumerate(target) if
                    i in idx]
            bbox = torch.cat(bbox, dim=0).float()
            inst_labels = torch.tensor([inst_labels[i] for i in idx], dtype=torch.int64)
            inst_areas = torch.tensor([inst_areas[i] for i in idx], dtype=torch.float32)

            # Check for invalid bounding boxes and replace with zeros
            invalid_mask = (bbox[:, 2] <= bbox[:, 0]) | (bbox[:, 3] <= bbox[:, 1])
            bbox[invalid_mask] = 0

        # Store in dictionary
        prepared_target['masks'] = inst_masks.float()
        prepared_target['transforms'] = inst_transforms.float()
        prepared_target['boxes'] = bbox
        prepared_target['area'] = inst_areas
        prepared_target['labels'] = inst_labels

        prepared_targets.append(prepared_target)

    return prepared_targets


def collate_fn(batch):
    """Custom collate function to prepare data for the model"""
    images, targets = zip(*batch)

    # Stack images on first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    # TODO Figure this out but for now choose 3 channels
    batched_images = batched_images[:, [0, 1, 2, 3, 4, 5, 6], :, :]
    # For some reason this needs to be under the 3 channel selection
    batched_images.requires_grad_(True)  # RCNN needs gradients on input images

    # Prepare targets
    prepared_targets = prepare_targets(targets)
    # prepared_targets = prepare_targets_heuristic(targets)

    # Return batch as a tuple
    return batched_images, prepared_targets


def createDataLoader():
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_TEST')
    ano_path = coco_path.joinpath('annotations')
    # Load the COCO dataset
    dataset = CustomCocoDetection(root=str(coco_path), annFile=str(ano_path.joinpath('merged_coco.json')))

    # Create a PyTorch dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5, collate_fn=collate_fn)

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
        print(i)
