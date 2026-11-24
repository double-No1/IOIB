"""Dataset helpers: MVTec-style corruption, bag splits, VisA/LOCO converters."""
import csv
import glob
import os
import random
import shutil

import numpy as np
from PIL import Image
import tqdm


def sorted_glob(pattern, recursive=True):
    """``glob`` then sort so filesystem readdir order cannot change a run."""
    return sorted(glob.iglob(pattern, recursive=recursive))


def sorted_listdir(path):
    return sorted(os.listdir(path))


def deterministic_shuffle(my_list, seed):
    local_random = random.Random(seed)
    local_random.shuffle(my_list)
    return my_list


def corrupt_dataset(
    category,
    num_negative_target=10,
    data_root="./datasets/MVTec",
    output_category="temp",
    seed=0,
):
    """Copy a category into a working folder and inject test anomalies into train/good.

    Anomalies copied from the test set are renamed ``<defect>_corrupted_<file>.png``
    so filter precision/recall can be measured. This is the overlapping protocol
    used in the paper: the same anomalies remain in the test set.
    """
    if data_root[-1] == "/":
        data_root = data_root[:-1]
    if os.path.exists(data_root + "/" + output_category):
        shutil.rmtree(data_root + "/" + output_category)
    os.makedirs(data_root + "/" + output_category, exist_ok=True)
    os.makedirs(data_root + "/" + output_category + "/ground_truth", exist_ok=True)
    os.makedirs(data_root + "/" + output_category + "/test/good", exist_ok=True)
    os.makedirs(data_root + "/" + output_category + "/train/good", exist_ok=True)

    num_negative = 0
    filenames = sorted_glob(data_root + "/" + category + "/test/" + "**/*.png")
    deterministic_shuffle(filenames, seed)
    if num_negative_target < 1.0:
        num_negative_target = int(
            num_negative_target * len(os.listdir(data_root + "/" + category + "/train/good"))
        )
    for filename in filenames:
        if ("good" not in filename.split("/")) and num_negative < num_negative_target:
            num_negative += 1
            shutil.copyfile(
                filename,
                data_root
                + "/"
                + output_category
                + "/train/good/"
                + filename.split("/")[-2]
                + "_corrupted_"
                + filename.split("/")[-1],
            )
    for filename in sorted_glob(
        data_root + "/" + category + "/train/good/**/*.png"
    ):
        os.makedirs(
            os.path.dirname(filename.replace(category, output_category)), exist_ok=True
        )
        shutil.copyfile(filename, filename.replace(category, output_category))
    for filename in sorted_glob(
        data_root + "/" + category + "/test/**/*.png"
    ):
        os.makedirs(
            os.path.dirname(filename.replace(category, output_category)), exist_ok=True
        )
        shutil.copyfile(filename, filename.replace(category, output_category))
    for filename in sorted_glob(
        data_root + "/" + category + "/ground_truth/**/*.png"
    ):
        os.makedirs(
            os.path.dirname(filename.replace(category, output_category)), exist_ok=True
        )
        shutil.copyfile(filename, filename.replace(category, output_category))
    if os.path.exists(data_root + "/" + category + "/validation/"):
        for filename in sorted_glob(
            data_root + "/" + category + "/validation/**/*.png"
        ):
            os.makedirs(
                os.path.dirname(filename.replace(category, output_category)), exist_ok=True
            )
            shutil.copyfile(filename, filename.replace(category, output_category))


def split_dataset(directory, category, folds):
    """Split ``train/good`` into ``folds`` non-overlapping bags (no replacement)."""
    for n in range(folds):
        if os.path.exists(directory.replace(category, category + "_" + str(n))):
            shutil.rmtree(directory.replace(category, category + "_" + str(n)))

    files = sorted_listdir(directory + "train/good/")
    random.shuffle(files)
    chunk_size = len(files) // folds
    chunks = [files[i * chunk_size : (i + 1) * chunk_size] for i in range(folds)]

    directories = []
    for n in range(folds):
        os.makedirs(
            directory.replace(category, category + "_" + str(n)) + "train/good/",
            exist_ok=True,
        )
        directories.append(directory.replace(category, category + "_" + str(n)))
        for file_name in chunks[n]:
            shutil.copyfile(
                directory + "train/good/" + file_name,
                directory.replace(category, category + "_" + str(n))
                + "train/good/"
                + file_name,
            )
        for filename in sorted_glob(directory + "/test/**/*.png"):
            os.makedirs(
                os.path.dirname(filename.replace(category, category + "_" + str(n))),
                exist_ok=True,
            )
            shutil.copyfile(
                filename, filename.replace(category, category + "_" + str(n))
            )
        for filename in sorted_glob(
            directory + "/ground_truth/**/*.png"
        ):
            os.makedirs(
                os.path.dirname(filename.replace(category, category + "_" + str(n))),
                exist_ok=True,
            )
            shutil.copyfile(
                filename, filename.replace(category, category + "_" + str(n))
            )
        if os.path.exists(directory + "/validation/"):
            for filename in sorted_glob(
                directory + "/validation/**/*.png"
            ):
                os.makedirs(
                    os.path.dirname(filename.replace(category, category + "_" + str(n))),
                    exist_ok=True,
                )
                shutil.copyfile(
                    filename, filename.replace(category, category + "_" + str(n))
                )
    return directories


def _mkdirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_visa(data_folder, save_folder, split_file):
    """Convert VisA to MVTec folder layout. Images are saved as PNG.

    Taken from https://www.tomomi-research.com/en/archives/3854
    """
    data_list = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]
    for data in data_list:
        train_folder = os.path.join(save_folder, data, "train")
        test_folder = os.path.join(save_folder, data, "test")
        mask_folder = os.path.join(save_folder, data, "ground_truth")
        _mkdirs_if_not_exists(os.path.join(train_folder, "good"))
        _mkdirs_if_not_exists(os.path.join(test_folder, "good"))
        _mkdirs_if_not_exists(os.path.join(test_folder, "bad"))
        _mkdirs_if_not_exists(os.path.join(mask_folder, "bad"))

    with open(split_file, "r") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            object_name, set_type, label, image_path, mask_path = row
            label = "good" if label == "normal" else "bad"
            img_src_path = os.path.join(data_folder, image_path)
            image = Image.open(img_src_path).convert("RGB")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            image_name = base_name + ".png"
            img_dst_path = os.path.join(
                save_folder, object_name, set_type, label, image_name
            )
            image.save(img_dst_path, format="PNG")
            if set_type == "test" and label == "bad":
                msk_src_path = os.path.join(data_folder, mask_path)
                mask_name = base_name + "_mask.png"
                msk_dst_path = os.path.join(
                    save_folder, object_name, "ground_truth", label, mask_name
                )
                mask = Image.open(msk_src_path)
                mask_array = np.array(mask)
                mask_array[mask_array != 0] = 255
                Image.fromarray(mask_array.astype("uint8")).save(msk_dst_path)


def convert_visa_to_mvtec_format(source=None, output=None):
    if source is None:
        visa_dir = list(glob.iglob("./**/pipe_fryum", recursive=True))
        if not len(visa_dir):
            print("Could not find VisA (looked for a pipe_fryum folder).")
            return None
        if len(visa_dir) > 1:
            print("Found multiple VisA datasets:")
            for directory in visa_dir:
                print(directory.split("pipe_fryum")[0])
            return None
        source = visa_dir[0].split("pipe_fryum")[0]
    if output is None:
        source_norm = source.rstrip("/")
        parent, name = os.path.split(source_norm)
        output = os.path.join(parent, name + "_MVTec_formated")
    split_candidates = sorted(
        glob.iglob(os.path.join(source, "**", "1cls.csv"), recursive=True)
    )
    if not split_candidates:
        split_candidates = list(glob.iglob("./**/1cls.csv", recursive=True))
    if not split_candidates:
        print("Could not find VisA split file 1cls.csv.")
        return None
    os.makedirs(output, exist_ok=True)
    print(f"Writing VisA (MVTec layout) to {output}")
    prepare_visa(source, output, split_candidates[0])
    return output


def convert_loco_to_mvtec_format(source=None, output=None):
    if source is None:
        loco_dir = list(glob.iglob("./**/splicing_connectors", recursive=True))
        if not len(loco_dir):
            print("Could not find MVTec LOCO AD (looked for splicing_connectors).")
            return None
        if len(loco_dir) > 1:
            print("Found multiple LOCO AD datasets:")
            for directory in loco_dir:
                print(directory.split("splicing_connectors")[0])
            return None
        source = loco_dir[0].split("splicing_connectors")[0]
    if output is None:
        source_norm = source.rstrip("/")
        parent, name = os.path.split(source_norm)
        output = os.path.join(parent, name + "_MVTec_formated")
    print(f"Writing LOCO AD (MVTec layout) to {output}")
    file_names = sorted_glob(os.path.join(source, "**", "*.png"))
    for file_name in tqdm.tqdm(file_names):
        if "ground_truth" not in file_name.split("/"):
            dest = file_name.replace(source, output)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(file_name, dest)
        else:
            gt_number = file_name.split("/")[-2]
            anomaly_number = file_name.split("/")[-1].replace(".png", "")
            target_output = file_name.replace(source, output).replace(
                gt_number + "/" + anomaly_number + ".png", gt_number + "_mask.png"
            )
            if not os.path.exists(target_output):
                os.makedirs(os.path.dirname(target_output), exist_ok=True)
                shutil.copyfile(file_name, target_output)
            else:
                source_img = np.array(Image.open(file_name).convert("RGB"))
                existing = np.array(Image.open(target_output).convert("RGB"))
                Image.fromarray((source_img + existing).astype("uint8")).save(
                    target_output
                )
    return output
