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
    inst_masks = torch.nn.utils.rnn.pad_sequence(inst_masks, batch_first=True, padding_value=0)

    inst_masks = inst_masks.type(torch.bool)  # binary masks

    return inst_masks, inst_labels


def prepare_targets(targets) -> list:
    prepared_targets = []
    for target in targets:
        prepared_target = {}
        # Masks and mask labels
        inst_masks, inst_labels = prepare_masks(target)

        # Transformation Matrices
        inst_transforms = [torch.tensor(inst['transform']) for inst in target]
        # turn them into np arrays
        inst_transforms = [np.array(inst_transform) for inst_transform in inst_transforms]
        inst_transforms = [i.reshape((4, 4), order='F') for i in inst_transforms]
        inst_transforms = [torch.from_numpy(inst_transform) for inst_transform in inst_transforms]
        inst_transforms = torch.stack(inst_transforms, dim=0)
        inst_transforms = inst_transforms.type(torch.float32)

        if torch.isnan(inst_transforms).any() or torch.isinf(inst_transforms).any():
            print("Targets are nan or inf")
            print(targets)
            raise ValueError("Targets are nan or inf")
        # Store in dictionary
        prepared_target['masks'] = inst_masks.float()
        prepared_target['tm'] = inst_transforms.float()
        # labels need to be int64, such overkill
        prepared_target['labels'] = torch.tensor(inst_labels, dtype=torch.int64)
        prepared_target['names'] = np.array([np.array(inst['name']) for inst in target])

        prepared_targets.append(prepared_target)

    # List of dictionaries - one dictionary per image
    return prepared_targets


def homog_to_trans_and_axis_angle(homog):
    batchsize = homog.shape[0]
    R = homog[:, :3, :3]
    t = homog[:, :3, 3]
    r = Rotation.from_matrix(R)
    axis_angle = r.as_rotvec()
    return t, torch.Tensor(axis_angle)


def collate_TM_GT(batch, channels=None):
    """Custom collate function to prepare data for the model"""
    images, targets = zip(*batch)

    # Stack images on first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    del images
    # Select channels to use
    if channels:
        batched_images = batched_images[:, channels, :, :]

    # Prepare targets
    prepared_targets = prepare_targets(targets)
    del targets

    # Will need to turn image + instances into a microbatch but this is better done in the training loop
    # because of vram limitations

    # strip the first 2 entries in everything - backround and bin - not needed possibly break stuff
    masks = [prepared_target['masks'][2:] for prepared_target in prepared_targets]

    # Transform matrices into translation and axis angle rotation representation
    tms = [prepared_target['tm'][2:] for prepared_target in prepared_targets]

    labels = [prepared_target['labels'][2:] for prepared_target in prepared_targets]
    names = [prepared_target['names'][2:] for prepared_target in prepared_targets]

    # check if the targets are empty
    empty = []
    for i in range(len(labels)):
        if len(labels[i]) == 0:
            empty.append(i)
    # remove the empty targets
    for i in sorted(empty, reverse=True):
        del masks[i], tms[i], labels[i], names[i],
        # tensor doesn't have a del function
        # gotta do it like this
        batched_images = torch.cat((batched_images[:i], batched_images[i + 1:]), 0)

    masks, tms, labels, names = permute_microbatch(masks, tms, labels, names)

    translation, axis_angle = zip(*[homog_to_trans_and_axis_angle(tm) for tm in tms])

    # scale the tranlation to -1, 1 range and substract 600 from z

    # Return batch as a tuple
    return batched_images, masks, axis_angle, translation, names


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
