"""Bootstrap Aggregation Anomaly Filtering (Algorithm 1).
"""
import gc
import json
import math
import os
import random
import shutil
from collections import Counter

import torch

from baaf.data import corrupt_dataset, split_dataset
from baaf.determinism import configure_determinism
from baaf.gmm import dual_gaussian_thresh
from baaf.metrics import do_logging


def super_seed(seed):
    """Seed Python, NumPy, and PyTorch and enable deterministic CUDA kernels."""
    configure_determinism(seed, warn_only=True)


def run_baaf(
    category,
    method,
    data_dir="./datasets/MVTecAD/",
    seed=42,
    k=2,
    votes=3,
    num_per_negative=0.1,
    output_dir="./results/",
    save_qualitative_results=False,
    evaluate_final_model=True,
    save_gmm_plots=False,
    verbose=True,
):
    """Run BAAF(votes/k) + ``method`` on one dataset category.

    Parameters match the paper: ``k`` is the number of bags, ``votes`` is the
    number of independent bagging rounds. ``votes=0`` skips filtering and trains
    the OCC baseline on the (possibly corrupted) training set.
    """
    if data_dir[-1] != "/":
        data_dir = data_dir + "/"
    os.makedirs(output_dir, exist_ok=True)

    super_seed(seed)
    corrupt_dataset(
        category,
        num_negative_target=num_per_negative,
        data_root=data_dir,
        seed=seed,
        output_category=category + "_" + type(method).__name__,
    )
    train_data_dir = data_dir + category + "_" + type(method).__name__ + "/"
    hold_outs = []
    hold_out_info = {"_detailed_folds": {}}

    for run in range(votes):
        fold_directories = split_dataset(
            train_data_dir, category + "_" + type(method).__name__, k
        )
        anomaly_predictions = {}
        for fold_index, fold_directory in enumerate(fold_directories):
            if verbose:
                print(run, fold_directory)
            method.reset()
            torch.cuda.empty_cache()
            gc.collect()
            method.train_on_class(random.randint(0, 1000), fold_directory)
            fold_train = set(os.listdir(fold_directory + "train/good/"))
            for file_name in sorted(os.listdir(train_data_dir + "train/good/")):
                if file_name not in fold_train:
                    img_pred, _ = method.predict_on_image(
                        train_data_dir + "train/good/" + file_name
                    )
                    if file_name not in anomaly_predictions:
                        anomaly_predictions[file_name] = [None] * len(fold_directories)
                    anomaly_predictions[file_name][fold_index] = img_pred
            if os.path.exists(fold_directory):
                shutil.rmtree(fold_directory)
            torch.cuda.empty_cache()
            gc.collect()

        # Faithful to the implementation that produced the paper numbers: only
        # the first k-1 bag models cast filter votes. The k-th model is still
        # trained (it keeps the RNG stream and per-bag seeds identical to the
        # published runs) but its predictions are not tallied. This is slightly
        # stricter than Algorithm 1 in the paper, where all bag models vote.
        for fold_ref in range(k - 1):
            single_anomaly_predictions = {
                key: value[fold_ref]
                for key, value in anomaly_predictions.items()
                if value[fold_ref] is not None
            }
            lo = min(single_anomaly_predictions.values())
            single_anomaly_predictions = {
                key: value - lo for key, value in single_anomaly_predictions.items()
            }
            hi = max(single_anomaly_predictions.values())
            single_anomaly_predictions = {
                key: value / hi for key, value in single_anomaly_predictions.items()
            }
            hold_out_info["_detailed_folds"][
                f"run_{run}_fold_{fold_ref}_predictions_on_folds"
            ] = single_anomaly_predictions
            plot_name = None
            if save_gmm_plots:
                plot_name = os.path.join(
                    output_dir,
                    f"{type(method).__name__}_{seed}_{category}_num_neg-{str(num_per_negative)}"
                    f"_votes-{str(votes)}_run-{str(run)}_fold-{str(fold_ref)}",
                )
            threshold = dual_gaussian_thresh(
                flat_preds=single_anomaly_predictions,
                plot_name=plot_name,
                verbose=verbose,
            )
            hold_out_info["_detailed_folds"][
                f"run_{run}_fold_{fold_ref}_threshold"
            ] = threshold
            fold_hold_outs = sorted(
                key_value
                for key_value in single_anomaly_predictions
                if single_anomaly_predictions[key_value] >= threshold
            )
            hold_outs += fold_hold_outs
            hold_out_info["_detailed_folds"][
                f"run_{run}_fold_{fold_ref}_hold_outs"
            ] = fold_hold_outs

    train_images_list = sorted(os.listdir(train_data_dir + "train/good/"))
    vote_counts = Counter(hold_outs)
    final_hold_outs = [
        file_name
        for file_name in train_images_list
        if vote_counts[file_name] > math.floor(((k - 1) * votes) / 2.0)
    ]

    result_path, results = None, None
    out_dir = output_dir if output_dir.endswith("/") else output_dir + "/"
    if evaluate_final_model:
        result_path, results = do_logging(
            train_images_list=train_images_list,
            final_hold_outs=final_hold_outs,
            hold_out_info=hold_out_info,
            category=category,
            method=method,
            seed=seed,
            k=k,
            votes=votes,
            num_per_negative=num_per_negative,
            data_dir=data_dir,
            output_dir=out_dir,
            save_qualitative_results=save_qualitative_results,
            train_data_dir=train_data_dir,
        )
    else:
        os.makedirs(out_dir, exist_ok=True)
        hold_out_info["_detailed_folds"]["final_hold_outs"] = final_hold_outs
        result_path = os.path.join(
            out_dir,
            f"{type(method).__name__}_{seed}_{category}_num_neg_{str(num_per_negative)}"
            f"_votes{str(votes)}_folds{str(k)}_.json",
        )
        results = {"hold_out_info": hold_out_info}
        with open(result_path, "w") as file:
            json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))

    if os.path.exists(train_data_dir):
        shutil.rmtree(train_data_dir)
    return method, result_path, results
