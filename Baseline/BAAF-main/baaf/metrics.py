"""Masks are resized to 256 then center-cropped to 224."""
import glob
import json
import os
from bisect import bisect

import cv2
import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from sklearn import metrics
from scipy.ndimage import label

from baaf.determinism import environment_report
from baaf.viz import visualize_confidence


def do_logging(
    train_images_list,
    final_hold_outs,
    hold_out_info,
    category,
    method,
    seed,
    k,
    votes,
    num_per_negative,
    data_dir,
    output_dir,
    save_qualitative_results,
    train_data_dir,
):
    tp = fp = fn = tn = 0
    for file_name in train_images_list:
        if file_name in final_hold_outs:
            if "_corrupted_" in file_name:
                tp += 1
            else:
                fp += 1
        else:
            if "_corrupted_" in file_name:
                fn += 1
            else:
                tn += 1

    for hold_out in final_hold_outs:
        os.remove(train_data_dir + "train/good/" + hold_out)

    method.reset()
    method.train_on_class(seed, train_data_dir)

    img_results, pixel_results = do_evaluation(
        category + "_" + type(method).__name__,
        method,
        votes=votes,
        folds=k,
        data_dir=data_dir,
        save_qualitative_results=save_qualitative_results,
    )
    img_results["threshold"] = []

    hold_out_info["_detailed_folds"]["final_hold_outs"] = final_hold_outs
    hold_out_info["final_hold_outs_precision"] = tp / (tp + fp) if tp + fp else tp
    hold_out_info["final_hold_outs_recall"] = tp / (tp + fn) if tp + fn else tp
    hold_out_info["final_hold_outs_tp"] = tp
    hold_out_info["final_hold_outs_fp"] = fp
    hold_out_info["final_hold_outs_fn"] = fn
    hold_out_info["final_hold_outs_tn"] = tn

    try:
        import git

        repo = git.Repo(search_parent_directories=True).head.object.hexsha
    except Exception:
        repo = "git info not found"

    final_results = {
        "hold_out_info": hold_out_info,
        "img_results": img_results,
        "pixel_results": pixel_results,
        "code_version_git_hash": repo,
        "cuda_version": str(torch.version.cuda),
        "environment": environment_report(),
    }
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"{type(method).__name__}_{seed}_{category}_num_neg_{str(num_per_negative)}"
        f"_votes{str(votes)}_folds{str(k)}_.json",
    )
    with open(out_path, "w") as file:
        json.dump(final_results, file, sort_keys=True, indent=4, separators=(",", ": "))
    return out_path, final_results


def do_evaluation(
    category,
    method,
    votes,
    folds,
    data_dir="./datasets/MVTecAD/",
    save_qualitative_results=False,
):
    test_mask_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.0, std=1.0),
        ]
    )
    test_images = sorted(
        glob.iglob(data_dir + category + "/test/" + "**/*.png", recursive=True)
    )
    test_masks = sorted(
        glob.iglob(data_dir + category + "/ground_truth/" + "**/*.png", recursive=True)
    )

    ground_truth_masks, ground_truth_imgs = [], []
    predictions_img, predictions_maps = [], []
    raw_maps = []

    # Single inference pass: raw maps are kept so the min/max used to normalize
    # qualitative overlays can be computed without predicting twice.
    for test_image_path in test_images:
        tp = os.path.normpath(test_image_path)
        defect_class = os.path.basename(os.path.dirname(tp))
        cat_root = os.path.dirname(os.path.dirname(os.path.dirname(tp)))
        flat_candidate = os.path.join(cat_root, "ground_truth", f"{defect_class}_mask.png")
        if os.path.isfile(flat_candidate):
            test_mask_path = [flat_candidate]
        else:
            test_mask_path = [
                text
                for text in test_masks
                if (
                    test_image_path.split("/")[-2]
                    + "/"
                    + test_image_path.split("/")[-1].replace(".", "_mask.")
                    in text.replace("\\", "/")
                )
            ]

        if len(test_mask_path) == 0:
            gt_mask = np.zeros((1, 224, 224))
        else:
            gt_mask = PIL.Image.open(test_mask_path[0])
            gt_mask = test_mask_transform(gt_mask)
            gt_mask = gt_mask[0:1]
            gt_mask = np.round(gt_mask.numpy())

        ground_truth_masks.append(gt_mask)
        ground_truth_imgs.append(gt_mask.max())

        img_prediction, map_pred = method.predict_on_image(test_image_path)
        predictions_img.append(img_prediction)
        raw_maps.append(map_pred)

    min_norm, max_norm = 1e10, 1e-10
    for map_pred in raw_maps:
        min_norm = min(min_norm, np.min(map_pred))
        max_norm = max(max_norm, np.max(map_pred))

    for test_image_path, map_pred, gt_mask in zip(test_images, raw_maps, ground_truth_masks):
        map_pred = (map_pred - min_norm) / (max_norm - min_norm)

        if save_qualitative_results:
            qual_dir = os.path.join(
                "./qualitative",
                type(method).__name__ + "_" + str(votes) + "_" + str(folds),
                category,
            )
            os.makedirs(qual_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(
                    qual_dir,
                    "qual_"
                    + test_image_path.split("/")[-2]
                    + "_"
                    + test_image_path.split("/")[-1],
                ),
                visualize_confidence(
                    cv2.imread(test_image_path),
                    map_pred[0, :, :],
                    truth=gt_mask,
                    display_size=(512, 512),
                ),
            )

        predictions_maps.append(map_pred)

    ground_truth_masks = np.array(ground_truth_masks)
    ground_truth_imgs = np.array(ground_truth_imgs)
    predictions_img = np.array(predictions_img)
    predictions_maps = np.array(predictions_maps)

    predictions_img = predictions_img - np.min(predictions_img)
    predictions_img = predictions_img / np.max(predictions_img)
    predictions_maps = predictions_maps - np.min(predictions_maps)
    predictions_maps = predictions_maps / np.max(predictions_maps)

    img_results = compute_imagewise_retrieval_metrics(predictions_img, ground_truth_imgs)
    pixel_results = compute_pixelwise_retrieval_metrics(predictions_maps, ground_truth_masks)
    pixel_results["aupro_score"] = compute_pro(
        predictions_maps[:, 0, :, :], ground_truth_masks[:, 0, :, :]
    )
    img_results["fpr"] = list(img_results["fpr"])
    img_results["tpr"] = list(img_results["tpr"])
    pixel_results["fpr"] = "Ignored"
    pixel_results["tpr"] = "Ignored"
    pixel_results["optimal_threshold"] = float(pixel_results["optimal_threshold"])
    return img_results, pixel_results


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()
    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)
    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def trapezoid(x, y, x_max=None):
    """MVTec official PRO integration helper."""
    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            "WARNING: Not all x and y values passed to trapezoid(...) "
            "are finite. Will continue with only the finite values."
        )
    x = x[finite_mask]
    y = y[finite_mask]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_interp = y[ins - 1] + (
                (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            )
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def compute_pro(anomaly_maps, ground_truth_maps):
    """Official MVTec PRO (https://www.mvtec.com/company/research/datasets/mvtec-ad)."""
    structure = np.ones((3, 3), dtype=int)
    num_ok_pixels = 0
    num_gt_regions = 0
    shape = (len(anomaly_maps), anomaly_maps[0].shape[0], anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max
    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components
        ok_mask = labeled == 0
        num_ok_pixels += np.sum(ok_mask)
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1.0 / region_size
        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    del sort_idxs

    np.cumsum(fp_changes_flat, out=fp_changes_flat)
    fp_changes_flat = fp_changes_flat.astype(np.float32, copy=False)
    np.divide(fp_changes_flat, num_ok_pixels, out=fp_changes_flat)
    np.cumsum(pro_changes_flat, out=pro_changes_flat)
    np.divide(pro_changes_flat, num_gt_regions, out=pro_changes_flat)

    keep_mask = np.append(np.diff(anomaly_scores_flat) != 0, np.True_)
    del anomaly_scores_flat
    fprs = fp_changes_flat[keep_mask]
    pros = pro_changes_flat[keep_mask]
    del keep_mask
    np.clip(fprs, a_min=None, a_max=1.0, out=fprs)
    np.clip(pros, a_min=None, a_max=1.0, out=pros)
    zero, one = np.array([0.0]), np.array([1.0])
    return trapezoid(np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one)), x_max=0.3) / 0.3
