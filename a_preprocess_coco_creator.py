import json
import numpy as np
import pycocotools.mask as mask_util


def create_image_json(rgbd, image_id):
    image = {}
    image["id"] = image_id
    image["width"] = rgbd.shape[1]
    image["height"] = rgbd.shape[0]
    image["file_name"] = "train" + "/" + str(image_id) + ".png"
    image["license"] = 1
    return image


def create_annotations_json(labels, labels_info, tm, image_id, bin_transform, categories):
    annotations = []
    uniq = np.unique(labels.flatten())
    # get a reverse dictionary for the labels
    labels_info = {v: k for k, v_list in labels_info.items() for v in v_list}
    for i, id in enumerate(uniq):
        anotation = {}
        # get the global id of the label - the labels labels_info are local to the scan
        global_id = categories.where(labels_info[id])
        anotation["id"] = global_id  # global id of the label
        # label name is the same in local and global context
        anotation["name"] = labels_info[id]  # human-readable name of the label
        anotation["image_id"] = image_id  # id of the image the label is in used for matching to image

        # create the binary mask for the specific label
        mask = (labels == id).astype(np.bool)
        # encode the mask with run length encoding
        rle = mask_util.encode(np.asfortranarray(mask))

        # add the rle to the annotation
        anotation["segmentation"] = rle
        anotation["area"] = mask_util.area(rle)
        anotation["bbox"] = mask_util.toBbox(rle)  # here just so I don't have to change the network

        # use the local id to identify which transform to use
        if id == 0:
            # background doesn't have a transform
            anotation["transform"] = np.eye(4)
            anotation["category_id"] = 0
        elif id == 1:
            # bin transform is the same for all labels - changes only on a per-scan basis
            anotation["category_id"] = 1
            anotation["transform"] = bin_transform
        else:
            # the rest of the labels have their own  transform
            anotation["transform"] = tm[id - 2]
            anotation["category_id"] = 2


        # anotation["supercategory"] = labels_info[id][0] # not needed
        # ann["iscrowd"] = 0 # not needed
        annotations.append(anotation)

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
        anotation["transform"] = anotation["transform"].tolist()
        anotation["bbox"] = anotation["bbox"].tolist()
    return image, anotations


def create_json(results, categories, save_path):
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

    # save the json file
    with open(save_path.joinpath('annotations', 'coco.json'), 'w') as f:
        json.dump(coco, f)