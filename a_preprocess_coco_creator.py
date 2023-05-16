import json

import numpy as np
import pycocotools.mask as mask_util

from a_preprocess_utils import is_mask_good


def create_image_json(shape, image_id):
    image = {}
    image["id"] = image_id
    image["width"] = shape[1]
    image["height"] = shape[0]
    image["file_name"] = "train" + "/" + str(image_id) + ".npz"
    image["license"] = 1
    return image


def create_annotations_json(labels, labels_info, tm, image_id, bin_transform, categories, inputs, stats, stls):
    annotations = []
    uniq = np.unique(labels.flatten())

    # count number of bins in the self categories
    num_bins = sum([1 for x in categories if "bin" in x])

    # get a reverse dictionary for the labels
    labels_info = {v: k for k, v_list in labels_info.items() for v in v_list}
    for i, id in enumerate(uniq):
        anotation = {}
        # get the global id of the label - the labels labels_info are local to the scan
        global_id = categories.where(labels_info[id])
        anotation["id"] = global_id  # local id of the label
        # label name is the same in local and global context
        anotation["name"] = labels_info[id]  # human-readable name of the label
        anotation["image_id"] = image_id  # id of the image the label is in used for matching to image

        # create the binary mask for the specific label
        mask = (labels == id).astype(np.bool)
        # encode the mask with run length encoding
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.bool)))

        # decide if the mask is good or not

        if id < 2 or is_mask_good(mask, inputs, labels_info[id], stls, tm[i - 2]):

            # add the rle to the annotation
            anotation["segmentation"] = rle
            anotation["area"] = mask_util.area(rle)
            anotation["bbox"] = mask_util.toBbox(rle)  # here just so I don't have to change the network

            # use the local id to identify which transform to use
            if id == 0:
                # background doesn't have a transform
                anotation["transform"] = [float(i) for i in np.eye(4).flatten().tolist()]
                anotation["category_id"] = 0
            elif id == 1:
                # bin transform is the same for all labels - changes only on a per-scan basis
                anotation["category_id"] = 1
                anotation["transform"] = [float(i) for i in bin_transform]
            else:
                # the rest of the labels have their own  transform
                anotation["transform"] = [float(i) for i in tm[i - 2]]
                anotation["category_id"] = global_id

            # anotation["supercategory"] = labels_info[id][0] # not needed
            # TODO consider adding the following for instances where area is less than x
            # ann["iscrowd"] = 0 # not needed
            annotations.append(anotation)
            # box = anotation["bbox"]
            # # ploat masks and bbox
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            # ax.imshow(mask)
            # ax.add_patch(plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='red', linewidth=3))
            # plt.show()

    # after testing, it appears that this annotations array needs to be flattened to be in the correct format
    # but more things depend on this format, so I'll do it when writing the json file
    return annotations


def prepare_data(image, anotations):
    """Prepare the data for the json file as it can take about 3 data types"""
    # image
    image["width"] = int(image["width"])
    image["height"] = int(image["height"])
    image["id"] = int(image["id"])
    image["license"] = int(image["license"])
    # anotations
    for anotation in anotations:
        anotation["id"] = int(anotation["id"])
        anotation["image_id"] = int(anotation["image_id"])
        anotation["category_id"] = int(anotation["category_id"])
        anotation["area"] = float(anotation["area"])
        anotation["segmentation"] = str(anotation["segmentation"])
        anotation["transform"] = list(anotation["transform"])
        anotation["bbox"] = anotation["bbox"].tolist()
    return image, anotations


def create_json(results, categories, save_path, mean, std, i):
    # read the base json file
    with open(save_path.joinpath('coco_base.json'), 'r') as f:
        coco = json.load(f)
    # add the images
    coco["images"] = [result[0] for result in results]

    # add the annotations
    # this amazing list comprehetion flattens the list of lists of annotations which is in results[1]
    annotations = [item for sublist in [result[1] for result in results] for item in sublist]
    coco["annotations"] = annotations

    # add the categories
    coco["categories"] = [{"id": i, "name": cat} for i, cat in enumerate(categories)]

    coco["image_stats"] = {"mean": [float(i) for i in np.mean(mean, axis=0).astype(float).tolist()],
                           "std": [float(i) for i in np.mean(std, axis=0).astype(float).tolist()]}

    # save the json file
    with open(save_path.joinpath(f'annotations', f'coco{i}.json'), 'w') as f:
        json.dump(coco, f)
