import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score
from scipy.optimize import linear_sum_assignment
import pandas as pd


def iou_score(output, target):
    # Ensure the masks are in boolean format
    output = np.asarray(output, dtype=bool)
    target = np.asarray(target, dtype=bool)

    intersection = (output & target).sum()
    union = (output | target).sum()

    return float(intersection) / float(union + 1e-6)


import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(dataloader, model, device, topN_range=(0, 6), blacklist=[0, 1, 2]):
    model = model.to(device)
    model.eval()

    matches = []

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for target, prediction in zip(targets, predictions):
                valid_indices = [i for i, label in enumerate(target['labels'].cpu().numpy()) if label not in blacklist]
                if not valid_indices:
                    continue

                gt_mask = target['masks'][valid_indices].cpu().numpy()

                valid_pred_indices = [i for i, label in enumerate(prediction['labels'].cpu().numpy()) if label not in blacklist]
                if not valid_pred_indices:
                    continue

                sorted_scores, sorted_indices = torch.sort(prediction['scores'][valid_pred_indices], descending=True)
                pred_mask = prediction['masks'][sorted_indices].squeeze(1).cpu().numpy()

                iou_matrix = np.zeros((gt_mask.shape[0], pred_mask.shape[0]))
                for i, gt in enumerate(gt_mask):
                    for j, pred in enumerate(pred_mask):
                        iou_matrix[i, j] = iou_score(gt, pred)

                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_gts = matched_indices[0]
                matched_preds = matched_indices[1]

                for gt_idx, pred_idx in zip(matched_gts, matched_preds):
                    gt_match = gt_mask[gt_idx]
                    pred_match = (pred_mask[pred_idx] > 0.5).astype(np.uint8)
                    score = sorted_scores[pred_idx].item()

                    matches.append((gt_match, pred_match, score))

    matches.sort(key=lambda x: x[2], reverse=True)  # sort all matches by score in descending order

    iou_list_topN = []
    mAP_list_topN = []
    AR_list_topN = []

    for topN in range(*topN_range):
        iou_list = []
        recall_list = []
        precision_list = []

        if topN == 0:
            continue

        for i in range(min(topN, len(matches))):
            gt_match, pred_match, _ = matches[i]

            iou = iou_score(gt_match, pred_match)
            recall = recall_score(gt_match.flatten(), pred_match.flatten())
            precision = precision_score(gt_match.flatten(), pred_match.flatten())

            iou_list.append(iou)
            recall_list.append(recall)
            precision_list.append(precision)

        mAP = np.mean(precision_list)
        AR = np.mean(recall_list)
        mean_iou = np.mean(iou_list)

        iou_list_topN.append(mean_iou)
        mAP_list_topN.append(mAP)
        AR_list_topN.append(AR)

        print(f'For topN = {topN}: Mean IoU: {mean_iou}, mAP: {mAP}, AR: {AR}')

    df = pd.DataFrame({
        'topN': range(1, topN_range[1]),
        'Mean IoU': iou_list_topN,
        'mAP': mAP_list_topN,
        'AR': AR_list_topN
    })

    # Set up the plot
    fig, ax = plt.subplots()

    # Set bar width
    bar_width = 0.25

    # Position of bars on x axis
    r1 = np.arange(len(df['Mean IoU']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Colorblind-friendly colors and edge color to clearly separate the bars
    colors = ['#377eb8', '#e41a1c', '#4daf4a']
    ax.bar(r1, df['Mean IoU'], color=colors[0], width=bar_width, edgecolor='darkgrey', label='Mean IoU')
    ax.bar(r2, df['mAP'], color=colors[1], width=bar_width, edgecolor='darkgrey', label='mAP')
    ax.bar(r3, df['AR'], color=colors[2], width=bar_width, edgecolor='darkgrey', label='AR')

    # Add xticks in the middle of the grouped bars
    ax.set_xlabel('Number of Detections', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_xticks([r + bar_width for r in range(len(df['Mean IoU']))])
    ax.set_xticklabels(df['topN'])

    ax.legend(frameon=False, fontsize=10)

    ax.set_title('Evaluation Metrics for Different topN Detections', fontsize=14)

    # Removet spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Display grid lines behind the bars
    ax.set_axisbelow(True)

    # Increase the plot size
    fig.set_size_inches(10, 6)

    plt.show()

    return matches




if __name__ == '__main__':
    from b_DataLoader_RCNN import createDataLoader
    from c_MaskRCNN import MaskRCNN
    import pathlib

    base_path = pathlib.Path(__file__).parent.absolute()
    coco_path = base_path.joinpath('Test')
    chans = [0, 1, 2, 5, 9]

    loader, stats = createDataLoader(coco_path, bs=6, num_workers=4, channels=chans, shuffle=False,
                                     dataset_creation=True, anoname="coco.json")
    dataset = loader.dataset
    mean, std = stats
    mean, std = np.array(mean)[chans], np.array(std)[chans]

    model = MaskRCNN(num_classes=len(dataset.coco.cats), in_channels=5, mean=mean, std=std)
    checkpoint = "RCNN_Unscaled_2cat24.pth"
    model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    evaluate_model(loader, model, 'cuda')
