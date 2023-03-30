import os
import pathlib

import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.utils.data import DataLoader, Subset, random_split
from scipy.spatial.transform import Rotation


# Custom loader is needed to load RGB-A images - (4 channels) which in my case are RGB-D images
class CustomCocoDetection(CocoDetection):

    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile, transforms)
        # get image mean and std from the coco dataset, if not present, assume the default values (3 channel)
        image_stats = self.coco.dataset.get("image_stats", {})
        self.mean = np.array(image_stats.get("mean", [0.485, 0.456, 0.406]))
        self.std = np.array(image_stats.get("std", [0.229, 0.224, 0.225]))

    def _load_image(self, id: int) -> torch.Tensor:
        # overrride the _load_image method to load the images from the npz files in fp16
        # print("Loading image: ", id, "")
        path = self.coco.loadImgs(id)[0]["file_name"]
        # print("Path: ", path, " ")
        npz_file = np.load(os.path.join(self.root, path))
        # way to get the keys of the npz file, and load them as a list in the same order
        img_arrays = [npz_file[key] for key in npz_file.keys()]
        # dstack with numpy because pytorch refuses to stack different number of channels reasons
        image = np.dstack(img_arrays).astype(np.float32)
        # Channels are in the order of
        # R,G,B, X,Y,Z, NX,NY,NZ ,I
        image = torch.from_numpy(image).type(torch.float32)

        image_scaled = scale(image)
        # so the image is in the range of 0 to 1 not 0.5 to 1 as it is now
        # image_scaled[5] = (image_scaled[5] - 0.5) * 2
        # try:
        #     assert image_scaled[5].min() >= 0 and image_scaled[5].max() <= 1, "Image is not in the range of 0 to 1"
        # except AssertionError as e:
        #     print("Image min: ", image_scaled[5].min(), " max: ", image_scaled[5].max())
        #     raise e

        return image_scaled

    def _load_target(self, id: int):
        # override the _load_target becaiuse the original one is doing some weird stuff
        # no idea why it is doing that
        x = self.coco.imgToAnns[id]
        return x


def scale(image):
    # image appears to be -1 to 1
    # scale to 0 to 1
    image = (image + 1) / 2
    # print("Image min: ", image.min(), " max: ", image.max())
    return image


def prepare_masks(target):
    # Masks
    inst_masks = []
    inst_labels = []
    for inst in target:
        segmentation = eval(inst['segmentation'])

        decoded_mask = coco_mask.decode(segmentation)
        inst_mask = torch.from_numpy(decoded_mask).bool()

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

    # Pad masks to all be the same size, needed for torch to stack them
    try:
        inst_masks = torch.nn.utils.rnn.pad_sequence(inst_masks, batch_first=True, padding_value=0)
    except RuntimeError as e:
        print("Error: ", e)
        print("inst_masks: ", inst_masks)
        raise e


    inst_masks = inst_masks.type(torch.bool)  # binary masks

    return inst_masks, inst_labels


def prepare_transforms(target):
    # Extract transformation matrices from target
    inst_transforms = [torch.tensor(inst['transform']) for inst in target]
    # turn them into np arrays, so we can reshape them correctly
    inst_transforms = [np.array(inst_transform) for inst_transform in inst_transforms]
    # Order F because the matrices are in column major order (mathematicians ...)
    inst_transforms = [i.reshape((4, 4), order='F') for i in inst_transforms]
    # Extract rotation and translation
    inst_rotations = [inst_transform[:3, :3] for inst_transform in inst_transforms]
    # Extract translation - not actualy needed as target will use analytical methods from the predicted masks + lookup of depth
    inst_translations = [inst_transform[:3, 3] for inst_transform in inst_transforms]
    # turn them into torch tensors
    inst_rotations = [torch.from_numpy(inst_rotation).type(torch.float32) for inst_rotation in inst_rotations]
    inst_translations = [torch.from_numpy(inst_translation).type(torch.float32) for inst_translation in
                         inst_translations]
    # stack them into a single tensor - (N, 3, 3) dtype: torch.float32, N: number of instances (variable)
    inst_rotations = torch.stack(inst_rotations, dim=0)
    # stack them into a single tensor - (N, 3) dtype: torch.float32, N: number of instances (variable)
    inst_translations = torch.stack(inst_translations, dim=0)

    # now, the thinking goes like this:
    # Since the matrices are aligned on the axis with the infinite rotational symetry (Z axis)
    # We know that the rotation in the matrix will be aligned with the rotational symetry axis
    # So we can turn this problem into predicting the changed Z axis, in effect the change in coordinate system
    # This should basically dissregard the rotational symetry therefore symplifying the problem drastically
    # this line can be defined as point slope form with xyz being predicted / calculated from the mask
    # and the point being the center of mass of the mask
    # and slopes being predicted by the network

    return inst_rotations, inst_translations


def prepare_targets(targets) -> list:
    prepared_targets = []
    for target in targets:
        target = target[2:]
        if len(target) == 0:
            prepared_targets.append(None)
            continue
        prepared_target = {}
        # Masks and mask labels
        inst_masks, inst_labels = prepare_masks(target)
        # Transformations
        inst_rot, inst_move = prepare_transforms(target)

        # Store in dictionary
        prepared_target['masks'] = inst_masks.float()
        prepared_target['rot'] = inst_rot
        prepared_target['move'] = inst_move
        # labels need to be int64, such overkill
        prepared_target['labels'] = torch.tensor(inst_labels, dtype=torch.int64)
        prepared_target['names'] = np.array([np.array(inst['name']) for inst in target])

        bbox = [[inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
                 inst['bbox'][1] + inst['bbox'][3]] for inst in target]
        # bbox = torch.cat(bbox, dim=0)
        #
        # bbox = bbox.type(torch.float32)
        prepared_target['box'] = bbox

        prepared_targets.append(prepared_target)

    # List of dictionaries - one dictionary per image
    return prepared_targets


def prepare_batch(batch):
    images, targets = zip(*batch)

    # Stack images on first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    del images

    # Prepare targets
    prepared_targets = prepare_targets(targets)
    tocut = []
    for i in range(len(prepared_targets)):
        tocut.append(False if prepared_targets[i] is None else True)

    batched_images = batched_images[tocut]

    prepared_targets = [target for target in prepared_targets if target is not None]

    return batched_images, prepared_targets


def collate_TM_GT(batch, channels=None):
    """Custom collate function to prepare data for the model"""
    images, targets = prepare_batch(batch)

    if channels:
        images = images[:, channels, :, :]
    moves = [target['move'] for target in targets]
    # Get the Z translation for prediction as XY can be predicted from the mask
    zs = [move[:, 2] for move in moves]
    # zs = torch.stack(zs, dim=0)
    # XY seems to be usable from the PREDICTED mask because it has scoring, from the GT mask its bit more difficult
    # lets just use the TM gt for now
    xys = [move[:, :2] for move in moves]

    # Get the masks
    masks = [target['masks'] for target in targets]
    # Pad masks to all be the same size, needed for torch to stack them

    # Get the names
    names = [target['names'] for target in targets]
    # get rotation matrices
    rots = [target['rot'] for target in targets]

    # Extract the Z vector
    z_vecs = [rot[:, 2, :] for rot in rots]

    # change up the XY image to have the standard top left corner as origin
    # instead of the center of the image
    # this is needed for the crop to work correctly

    boxes = [target['box'] for target in targets]

    return images, masks, xys, zs, z_vecs, names, boxes


def permute_microbatch(masks, tms, labels, names):
    # permute the channels of the microbatch == permute the instances == permute the microbatch
    for idx, (mask, tm, label, name) in enumerate(zip(masks, tms, labels, names)):
        # permute the instances
        perm = torch.randperm(mask.shape[0])
        masks[idx] = mask[perm]
        tms[idx] = tm[perm]
        labels[idx] = label[perm]
        names[idx] = name[perm.tolist()]
    # names / labels should be predicted by the mask rcnn in the full pipeline
    return masks, tms, labels, names


class CollateWrapper:
    # Lambas are a nono in multiprocessing
    # need to be able to pickle the collate function to use it in multiprocessing, can't pickle lambdas
    def __init__(self, channels):
        self.collate_fn = collate_TM_GT
        self.channels = channels

    def __call__(self, batch):
        return self.collate_fn(batch, self.channels)


def createDataLoader(path, batchsize=1, shuffle=True, num_workers=4, channels: list = None, split=0.9):
    ano_path = (path.joinpath('annotations', "merged_coco.json"))

    collate = CollateWrapper(channels)

    # Load the COCO dataset
    dataset = CustomCocoDetection(root=str(path), annFile=str(ano_path))

    # Calculate the lengths of the train and validation sets
    train_len = int(len(dataset) * split)
    val_len = len(dataset) - train_len

    # Create the train and validation subsets
    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Create the PyTorch dataloaders for train and validation sets
    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers,
                                  collate_fn=collate)
    val_dataloader = DataLoader(val_set, batch_size=batchsize, shuffle=False, num_workers=num_workers,
                                collate_fn=collate)

    return train_dataloader, val_dataloader
