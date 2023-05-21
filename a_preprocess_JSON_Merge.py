import json
import glob
import numpy as np
from pathlib import Path

def is_valid_coco_json(coco_json):
    required_keys = {'info', 'categories', 'licenses', 'annotations', 'images', 'image_stats'}

    if not isinstance(coco_json, dict) or not required_keys.issubset(coco_json.keys()):
        return False

    if not isinstance(coco_json['annotations'], list) or not isinstance(coco_json['images'], list) or not isinstance(coco_json['categories'], list):
        return False

    return True

def validate_merged_coco_json(file_path):
    with open(file_path, 'r') as f:
        coco_json = json.load(f)

    if not is_valid_coco_json(coco_json):
        print("The merged JSON file is not a valid COCO JSON.")
        return

    num_images = len(coco_json['images'])
    num_annotations = len(coco_json['annotations'])
    num_categories = len(coco_json['categories'])

    print(f"The merged JSON file is valid and contains {num_images} images, {num_annotations} annotations, and {num_categories} categories.")

def merge_coco_jsons(json_files, save_path):
    """merge the jsons with end in their name with checking for duplicates"""
    merged_json = {}
    mean_list = []
    std_list = []
    idx = 0
    processed_images = {}

    for file in json_files:
        if 'end' not in file.lower() and "all" not in file.lower():
            continue
        print(f"Merging {file}...")
        with open(file, 'r') as f:
            coco_json = json.load(f)

        if not is_valid_coco_json(coco_json):
            print(f"Invalid COCO JSON format: {file}")
            continue

        if idx == 0:
            merged_json['info'] = coco_json['info']
            merged_json['categories'] = coco_json['categories']
            merged_json['licenses'] = coco_json['licenses']
            merged_json['annotations'] = []
            merged_json['images'] = []

        for image in coco_json['images']:
            if image['id'] not in processed_images:
                merged_json['images'].append(image)
                processed_images[image['id']] = image

        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id in processed_images:
                if annotation not in merged_json['annotations']:
                    merged_json['annotations'].append(annotation)

        mean_list.append(coco_json['image_stats']['mean'])
        std_list.append(coco_json['image_stats']['std'])

        idx += 1

    merged_json['image_stats'] = {
        'mean': np.mean(mean_list, axis=0).astype(float).tolist(),
        'std': np.mean(std_list, axis=0).astype(float).tolist()
    }

    with open(save_path, 'w') as f:
        json.dump(merged_json, f)


if __name__ == '__main__':
    abspath = Path(__file__).parent.absolute()
    coco_path = abspath.joinpath('RevertDSBIG')
    jsonspath = coco_path.joinpath('annotations')
    json_files = glob.glob(str(jsonspath.joinpath('*.json')))
    save_path = jsonspath.joinpath('merged.json')
    merge_coco_jsons(json_files, save_path)
    validate_merged_coco_json(save_path)
