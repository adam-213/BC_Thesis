import ray

from b_DataLoader_RCNN import createDataLoader
from b_MaskRCNN import MaskRCNN

import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import json
from pycocotools import mask as mask_util
from tqdm import tqdm
import multiprocessing as mp

intrinsics = {
    'fx': 1181.077335,
    'fy': 1181.077335,
    'cx': 516.0,
    'cy': 386.0
}


def plot(images, tmask, rle_mask, label, score, tlabel):
    # plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # convert the image back to the original form
    img = images[0].cpu().permute(1, 2, 0).detach().numpy()
    img = img.astype(np.uint8)
    img = img[:, :, :3]

    # plot the image with ground truth mask
    ax1.imshow(img)
    ax1.imshow(tmask, alpha=0.7)
    ax1.set_title(f"Ground Truth: Label {tlabel}")

    # plot the image with predicted mask
    masker = mask_util.decode(rle_mask)
    ax2.imshow(img)
    ax2.imshow(masker, alpha=0.7)
    ax2.set_title(f"Prediction: Label {label}, Score {score:.2f}")

    plt.show()


def world_to_image_coords(world_coords, intrinsics):
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    X, Y, Z = world_coords
    x = X / Z
    y = Y / Z
    u = fx * x + cx
    v = fy * y + cy
    u, v = round(u), round(v)
    return u, v


def geometric(ptc, mask, tensor=False):
    point_cloud = (ptc * mask).reshape(3, -1)
    valid_points = point_cloud[:, torch.any(point_cloud != 0, axis=0)]

    if valid_points.size(1) != 0:
        geometric_centroid = torch.mean(valid_points, axis=1)
        if tensor:
            return geometric_centroid
        else:
            return geometric_centroid[0].item(), geometric_centroid[1].item(), geometric_centroid[2].item()
    else:
        if tensor:
            return torch.tensor([float('inf'), float('inf'), float('inf')])
        else:
            return float('inf'), float('inf'), float('inf')


def compute_mask_iou(mask1, mask2):
    intersection = torch.sum((torch.bitwise_and(mask1.bool(), mask2.bool())).float())
    union = torch.sum((torch.bitwise_or(mask1.bool(), mask2.bool())).float())
    iou = intersection / union
    return iou.item()


def process_prediction(match, image_id, cats):
    # unpack the match
    centroid, mask, label, tm, score = match
    centroid_x, centroid_y = world_to_image_coords(centroid, intrinsics)

    # encodign and saving the new mask
    rle_mask = mask_util.encode(np.asfortranarray(mask.numpy().astype(bool)))
    rle_box = mask_util.toBbox(rle_mask)
    rle_area = mask_util.area(rle_mask)

    name = cats[label.item()]['name']
    if "bin" in name.lower() or "background" in name.lower() or "backround" in name.lower():
        return None

    annotation = {
        "id": int(label),
        "category_id": int(label),
        "image_id": int(image_id),
        "bbox": rle_box.tolist(),
        "score": float(score),
        "mask": rle_mask['counts'].decode('utf-8'),
        "size": rle_mask['size'],
        "area": float(rle_area),
        "transform": tm.flatten().tolist(),
        "name": name,
        "real_centroid": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
        "image_centroid": [float(centroid_x), float(centroid_y), float(centroid[2])],
    }

    return annotation


def compute_gt_centroid(idx, box_mask_label_score_tm, ptc):
    box, mask, label, tm = box_mask_label_score_tm
    real_centroid_x, real_centroid_y, depth_value = geometric(ptc, mask)

    return ((real_centroid_x, real_centroid_y, depth_value), mask, label, tm)


def compute_gt_centroids(ptc, target):
    # Prepare data for multiprocessing
    data = [(idx, (box, mask, label, tm), ptc) for idx, (box, mask, label, tm) in
            enumerate(zip(target['boxes'], target['masks'], target['labels'], target['tm']))]

    results = []
    for dat in data:
        results.append(compute_gt_centroid(*dat))
    # Doesn't seem to be needed
    # # Create a multiprocessing Pool
    # with mp.Pool(4) as pool:
    #     results = pool.starmap(compute_gt_centroid, data)

    return results


def compute_pre_centroid(idx, box_mask_label_score_tm, ptc):
    box, mask, label, score = box_mask_label_score_tm
    # threhold the mask
    mask = mask[0, :, :]
    mask = mask > 0.75
    real_centroid_x, real_centroid_y, depth_value = geometric(ptc, mask)

    return ((real_centroid_x, real_centroid_y, depth_value), mask, label, score)


def compute_pre_centroids(ptc, boxes, masks, labels, scores):
    # Prepare data for multiprocessing
    data = [(idx, (box, mask, label, score,), ptc) for idx, (box, mask, label, score) in
            enumerate(zip(boxes, masks, labels, scores))]

    results = []
    for dat in data:
        results.append(compute_pre_centroid(*dat))

    # # Create a multiprocessing Pool
    # with mp.Pool(4) as pool:
    #     results = pool.starmap(compute_pre_centroid, data)

    return results


def match(gt, pre):
    matched = []
    for i, (p_c, p_m, p_l, p_s) in enumerate(pre):
        best_mse = float('inf')
        best = None
        for j, (g_c, g_m, g_l, g_tm) in enumerate(gt):
            if g_l == p_l:
                iou = compute_mask_iou(p_m, g_m)
                # compute mse in all 3 dimensions
                mse = np.mean(np.square(np.subtract(p_c, g_c)))
                if iou > 0.85 and mse < best_mse:
                    best_mse = mse
                    best = (p_c, p_m, p_l, g_tm, p_s)

        if best is not None:
            matched.append(best)

    return matched


def main():
    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('known')
    channels = [0, 1, 2, 5, 9]
    full_loader, stats = createDataLoader(coco_path, bs=1, num_workers=0, channels=None, shuffle=False,
                                          # shuffle off is essential
                                          dataset_creation=True)  # otherwise everything is shuffled with images and annotations being mismatched
    mean, std = stats
    dataset = full_loader.dataset
    cats = full_loader.dataset.coco.cats
    model = MaskRCNN(5, len(cats), mean[channels], std[channels])

    checkpoint = torch.load("RCNN_Unscaled_19.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()

    coco_dict = {"images": [], "annotations": []}
    annotation_id = 1

    for idx, (fullimages, target) in enumerate(tqdm(full_loader)):
        gt = compute_gt_centroids(fullimages[0, 3:6, :, :], target[0])

        images = fullimages.clone()[:, channels, :, :]
        image_id = dataset.coco.imgs[idx]["id"]
        file_name = dataset.coco.imgs[image_id]["file_name"]
        assert int(file_name.strip("train/").strip(".npz")) == image_id == idx
        height = dataset.coco.imgs[image_id]["height"]
        width = dataset.coco.imgs[image_id]["width"]
        lic = dataset.coco.imgs[image_id]["license"]

        coco_dict["images"].append(
            {"id": image_id, "file_name": file_name, "height": height, "width": width, "license": lic})
        print(str(coco_path.joinpath("processed", f"{int(file_name.strip('train/').strip('.npz'))}.npz")))
        # np.savez_compressed(str(coco_path.joinpath("processed",f"{int(file_name.strip('train/').strip('.npz'))}.npz")),
        #                     fullimages=fullimages.cpu().detach().numpy())

        # plt.imshow(fullimages[0, 0:3, :, :].permute(1, 2, 0).cpu().detach())
        # for tm, label in zip(target[0]['tm'], target[0]['labels']):
        #     if label <= 2:
        #         continue
        #     tm = tm.cpu().detach().numpy().flatten(order='F')
        #     gtz = tm[[8, 9, 10]]
        #     gtc = tm[[12, 13, 14]]
        #     x, y = world_to_image_coords((gtc[0], gtc[1], gtc[2]), intrinsics)
        #     plt.scatter(x, y, c='r')
        #     scaling_factor = 100
        #     start = np.array([x, y])
        #     end = start + scaling_factor * np.array([gtz[0], gtz[1]])
        #
        #     points_hat_w = np.linspace(start, end, 100)
        #     for i in range(100 - 1):
        #         plt.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '+', color='g', alpha=0.5)
        #
        # #plt.savefig(f"i{idx}.png")
        # plt.show()
        # plt.close()
        # if idx > 100:
        #     return
        # continue

        images = images.cuda().float()
        outputs = model(images)

        boxes = outputs[0]['boxes'].cpu()
        labels = outputs[0]['labels'].cpu()
        scores = outputs[0]['scores'].cpu()
        masks = outputs[0]['masks'].cpu()

        pre = compute_pre_centroids(fullimages[0, 3:6, :, :], boxes, masks, labels, scores)

        matched = match(gt, pre)

        annotations = []
        for i, best in enumerate(matched):
            annotation = process_prediction(best, image_id, cats)
            if annotation:
                annotations.append(annotation)

            # # visualize
            # plt.imshow(fullimages[0, 0:3, :, :].permute(1, 2, 0).cpu().detach())
            # plt.imshow(best[1].cpu().detach(), alpha=0.5)
            # ctr = world_to_image_coords(best[0], intrinsics)
            # plt.scatter(ctr[0], ctr[1], c='r')
            # plt.title(f"{best[2]} - cats {cats[best[2].item()]['name']}")
            # # Project the 3D vector onto the 2D plane and scale by z-component
            #
            # tm = best[3].cpu().detach().numpy()
            # zdir = tm[:3, 2]
            # x,y = ctr[0], ctr[1]
            # scaling_factor = 100
            # start = np.array([x, y])
            # end = start + scaling_factor * np.array([zdir[0], zdir[1]])
            #
            # points_hat_w = np.linspace(start, end, 100)
            # for i in range(100 - 1):
            #     plt.plot(points_hat_w[i:i + 2, 0], points_hat_w[i:i + 2, 1], '+', color='g', alpha=0.5)
            #
            # # plt.savefig(f"i{idx}_{i}.png")
            # plt.show()
            # plt.close()

        coco_dict["annotations"].extend([a for a in annotations if a is not None])
        annotation_id += 1

        # delete everything that isnt in the annotations
        del images, outputs, boxes, labels, scores, masks, pre, matched, annotations
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with open(coco_path.joinpath("annotations", "merged.json"), 'r') as f:
        j = json.load(f)

    j['images'] = coco_dict['images']
    j['annotations'] = coco_dict['annotations']
    with open(coco_path.joinpath("annotations", "merged_maskrcnn_centroid.json"), 'w') as f:
        json.dump(j, f)


if __name__ == "__main__":
    # ray.init(num_cpus=16)
    main()
