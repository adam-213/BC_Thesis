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
        # get image mean and std from the coco dataset, if not present, assume the default values (3 channel)
        image_stats = self.coco.dataset.get("image_stats", {})
        self.mean = np.array(image_stats.get("mean", [0.485, 0.456, 0.406]))
        self.std = np.array(image_stats.get("std", [0.229, 0.224, 0.225]))

    def _load_image(self, id: int) -> torch.Tensor:
        # overrride the _load_image method to load the images from the npz files in fp16
        # print("Loading image: ", id, "")
        path = self.coco.loadImgs(id)[0]["file_name"]
        # print("Path: ", path, " ")
        # print("Path: ", path, " ")
        npz_file = np.load(os.path.join(self.root, path.replace("/", os.sep)))
        # way to get the keys of the npz file, and load them as a list in the same order
        img_arrays = [npz_file[key] for key in npz_file.keys()]
        # dstack with numpy because pytorch refuses to stack different number of channels reasons
        image = np.dstack(img_arrays).astype(np.float32)
        # Channels are in the order of
        # R,G,B, X,Y,Z, NX,NY,NZ ,I
        return torch.from_numpy(image).type(torch.float32)

    def _load_target(self, id: int):
        # override the _load_target becaiuse the original one is doing some weird stuff
        # no idea why it is doing that
        x = self.coco.imgToAnns[id]
        return x


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
        # inst_transforms = [i.reshape((4, 4), order='F') for i in inst_transforms]
        inst_transforms = [torch.from_numpy(inst_transform.flatten()) for inst_transform in inst_transforms]
        inst_transforms = torch.stack(inst_transforms, dim=0)
        inst_transforms = inst_transforms.type(torch.float32)

        # Bounding Boxes
        bbox = [torch.tensor([inst['bbox'][0], inst['bbox'][1], inst['bbox'][0] + inst['bbox'][2],
                              inst['bbox'][1] + inst['bbox'][3]]).view(1, 4) for inst in target]
        bbox = torch.cat(bbox, dim=0)

        bbox = bbox.type(torch.float32)

        # Store in dictionary
        prepared_target['masks'] = inst_masks.float()
        prepared_target['tm'] = inst_transforms.float()
        prepared_target['boxes'] = bbox.float()
        prepared_target['area'] = torch.tensor([inst['area'] for inst in target]).float()
        # labels need to be int64, such overkill
        prepared_target['labels'] = torch.tensor(inst_labels, dtype=torch.int64)

        # Don't know if necessary
        # prepared_target['image_id'] = torch.tensor([target[0]['image_id']])

        # Not defined - not needed
        # prepared_target['iscrowd'] = torch.zeros((bbox.shape[0],), dtype=torch.int8)

        prepared_targets.append(prepared_target)

    # List of dictionaries - one dictionary per image
    return prepared_targets


# Set up the camera intrinsic parameters
intrinsics = {
    'fx': 1181.077335,
    'fy': 1181.077335,
    'cx': 516.0,
    'cy': 386.0
}


def world_to_image_coords(world_coords, intrinsics):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    X, Y, Z = world_coords.numpy()
    # Normalize the real-world coordinates
    x = X / Z
    y = Y / Z
    # Apply the intrinsic parameters to convert to pixel coordinates
    u = fx * x + cx
    v = fy * y + cy
    # Round to integer values
    u, v = round(u), round(v)
    return u, v

import torch.nn.functional as F
def image_stack(images, targets):
    padding = (150, 150, 150, 150)
    new_images, new_targets = [], []
    for image, target in zip(images, targets):
        new_image, new_target = [], []
        target_masks = F.pad(target['masks'], padding, value=0)
        target_images = F.pad(image, padding, value=0)
        for inst in range(len(target['masks'])):
            if target['labels'][inst] <= 2:
                continue
            centroid = world_to_image_coords(target['tm'][inst, [12, 13, 14]], intrinsics)
            maskedimage = target_images * target_masks[inst, :, :]
            # pad the image by 150 on each side
            # crop to 224x224 on the centroid
            # assert centroid[0] < 1032 and centroid[1] < 772
            centroid = [centroid[0] + 150, centroid[1] + 150]

            maskedimage = maskedimage[:, centroid[1] - 112:centroid[1] + 112, centroid[0] - 112:centroid[0] + 112]

            new_image.append(torch.Tensor(maskedimage))
            new_target.append(torch.Tensor(target['tm'][inst, [8, 9, 10]]))

        if len(new_image) == 0 or len(new_target) == 0:
            continue
        new_targets.append(torch.stack(new_target))
        new_images.append(torch.stack(new_image))
    if len(new_images) == 0 or len(new_targets) == 0:
        return None, None
    # Stack images on first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.cat(new_images, dim=0)
    batched_targets = torch.cat(new_targets, dim=0)

    return batched_images, batched_targets


def collate_fn_rcnn(batch, channels=None, grad=True):
    """Custom collate function to prepare data for the model"""
    images, targets = zip(*batch)
    # Stack images on first dimension to get a tensor of shape (batch_size, C, H, W)
    batched_images = torch.stack(images, dim=0).permute(0, 3, 1, 2)

    prepared_targets = prepare_targets(targets)

    batched_images, batched_targets = image_stack(batched_images, prepared_targets)
    if batched_images is None or batched_targets is None:
        return None, None, None, None
    # Select channels to use
    if channels:
        batched_images = batched_images[:, channels, :, :]
    # For some reason this needs to be under the channel selection

    # Return batch as a tuple
    # gray the rgb images
    batched_images = torch.cat(
        (batched_images[:, 0:1, :, :] * 0.2989 + batched_images[:, 1:2, :, :] * 0.5870 + batched_images[:, 2:3, :,
                                                                                         :] * 0.1140,
         batched_images[:, 3:, :, :]), dim=1)

    cut = 75
    # if variable batch size is too big for the GPU, cut it down
    if batched_images.shape[0] > cut:
        batched_images = batched_images[:cut]
        batched_targets = batched_targets[:cut]

    return batched_images, batched_targets, None, None


from torch.utils.data import DataLoader, Subset, random_split


class CollateWrapper:
    # Lambas are a nono in multiprocessing
    # need to be able to pickle the collate function to use it in multiprocessing, can't pickle lambdas
    def __init__(self, channels, grad=True):
        self.collate_fn = collate_fn_rcnn
        self.channels = channels
        self.grad = grad

    def __call__(self, batch):
        return self.collate_fn(batch, self.channels, self.grad)


def createDataLoader(path, bs=1, shuffle=True, num_workers=6, channels: list = None, split=0.9,
                     dataset_creation=False, anoname='merged.json'):
    ano_path = (path.joinpath('annotations', anoname))

    # Load the COCO dataset
    dataset = CustomCocoDetection(root=str(path), annFile=str(ano_path))
    if dataset_creation:
        collate = CollateWrapper(channels, grad=False)
        loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=collate)
        return loader, (dataset.mean, dataset.std)
    else:
        collate = CollateWrapper(channels)
        # dataset = Subset(dataset, range(0, 100))
        # Calculate the lengths of the train and validation sets
        train_len = int(len(dataset) * split)
        val_len = len(dataset) - train_len

        # Create the train and validation subsets
        train_set, val_set = random_split(dataset,
                                          [train_len, val_len])  # this actually shuffles the dataset not just splits it
        # reason unknown, thankfully I noticed it

        # cut the dataset to a smaller size for testing
        # train_set = Subset(train_set, range(20))
        # val_set = Subset(val_set, range(10))

        # Create the PyTorch dataloaders for train and validation sets
        train_dataloader = DataLoader(train_set, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                                      collate_fn=collate)
        # shuffle=False for validation set to get the same results every time
        val_dataloader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate)
    # dataset =dataset.dataset
    return train_dataloader, val_dataloader, (dataset.mean, dataset.std)


def main():
    # iterate over the dataloader to get idea what is slow
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('COCO_Big')
    channels = [0, 1, 2, 5, 9]
    train_dataloader, val_dataloader, _ = createDataLoader(coco_path,
                                                           bs=3, shuffle=False, num_workers=0, channels=channels)

    for i, (images, targets, _, _) in enumerate(train_dataloader):
        print(i)
        print(images.shape)
        if i > 10:
            return

if __name__ == "__main__":
    main()

