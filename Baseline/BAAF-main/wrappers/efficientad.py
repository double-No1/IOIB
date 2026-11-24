"""EfficientAD wrapper for structural and logical anomaly detection.

The library in ``methods/EffecientAd`` is the unofficial nelson1425
reimplementation used in the paper, imported unmodified. Hyperparameters match
the research run: medium PDN, 256², 70k steps.
"""
import gc
import os
import tempfile
import urllib.request
import zipfile

import PIL
import torch
import tqdm
from torchvision import transforms

from baaf.determinism import configure_determinism, dataloader_generator, seed_worker
from wrappers.base import AD_Method_Wrapper

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_WEIGHTS = os.path.join(
    _ROOT, "methods", "EffecientAd", "models", "teacher_medium.pth"
)
_WEIGHTS_URL = (
    "https://github.com/open-edge-platform/anomalib/releases/download/"
    "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
)
# Anomalib checkpoints name conv layers; nelson1425 PDNs are nn.Sequential.
_ANOMALIB_TO_SEQUENTIAL = {
    "conv1": "0",
    "conv2": "3",
    "conv3": "6",
    "conv4": "8",
    "conv5": "10",
    "conv6": "12",
}


def sequential_teacher_state(state_dict):
    if not any(k.startswith("conv") for k in state_dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        prefix, suffix = key.split(".", 1)
        remapped[f"{_ANOMALIB_TO_SEQUENTIAL[prefix]}.{suffix}"] = value
    return remapped


def ensure_teacher_weights(path, model_size="medium"):
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading EfficientAD teacher weights to {path}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "weights.zip")
        urllib.request.urlretrieve(_WEIGHTS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        src = os.path.join(
            tmp, "efficientad_pretrained_weights", f"pretrained_teacher_{model_size}.pth"
        )
        if not os.path.isfile(src):
            matches = [
                os.path.join(dp, f)
                for dp, _, fs in os.walk(tmp)
                for f in fs
                if f.endswith(".pth") and model_size in f
            ]
            if not matches:
                raise FileNotFoundError(
                    f"Teacher weights not in the download archive. Place them at {path}."
                )
            src = matches[0]
        os.replace(src, path)
    return path


class EfficientADWrapper(AD_Method_Wrapper):
    def __init__(
        self,
        imagenet_train_path="./datasets/imagenette/imagenette2",
        weights=None,
        train_steps=70000,
        model_size="medium",
    ):
        self.out_channels = 384
        self.image_size = 256
        self.dataset = "mvtec_ad"
        self.model_size = model_size
        self.weights = weights or _DEFAULT_WEIGHTS
        self.imagenet_train_path = imagenet_train_path
        self.train_steps = train_steps

        self.default_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_ae = transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2),
                transforms.ColorJitter(saturation=0.2),
            ]
        )

        self.pretrain_penalty = True
        if self.imagenet_train_path == "none" or not os.path.isdir(self.imagenet_train_path):
            if self.imagenet_train_path not in (None, "none") and not os.path.isdir(
                self.imagenet_train_path
            ):
                print(
                    f"ImageNet/ImageNette not found at {self.imagenet_train_path}; "
                    "disabling the ImageNet penalty term."
                )
            self.pretrain_penalty = False

        self.reset()

    def _train_transform(self, image):
        return self.default_transform(image), self.default_transform(self.transform_ae(image))

    def train_on_class(self, seed, data_path):
        import itertools

        from torch.utils.data import DataLoader

        from methods.EffecientAd.common import (
            ImageFolderWithoutTarget,
            InfiniteDataloader,
            teacher_normalization,
        )

        # Reseed torch only: Python ``random`` drives later BAAF bag splits.
        configure_determinism(seed, warn_only=True, reseeds=("torch",))

        category = data_path.split("/")[-2]
        data_path = data_path.replace(category, "").replace("//", "/")

        full_train_set = ImageFolderWithoutTarget(
            os.path.join(data_path, category, "train"),
            transform=transforms.Lambda(self._train_transform),
        )

        if os.path.exists(os.path.join(data_path, category, "validation")):
            train_set = full_train_set
            validation_set = ImageFolderWithoutTarget(
                os.path.join(data_path, category, "validation"),
                transform=transforms.Lambda(self._train_transform),
            )
        else:
            train_size = int(0.9 * len(full_train_set))
            validation_size = len(full_train_set) - train_size
            rng = torch.Generator().manual_seed(seed)
            train_set, validation_set = torch.utils.data.random_split(
                full_train_set, [train_size, validation_size], rng
            )

        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=dataloader_generator(seed),
            worker_init_fn=seed_worker,
        )
        train_loader_infinite = InfiniteDataloader(train_loader)
        validation_loader = DataLoader(
            validation_set, batch_size=1, shuffle=False, num_workers=0
        )

        if self.pretrain_penalty:
            penalty_transform = transforms.Compose(
                [
                    transforms.Resize((2 * self.image_size, 2 * self.image_size)),
                    transforms.RandomGrayscale(0.3),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            penalty_set = ImageFolderWithoutTarget(
                self.imagenet_train_path, transform=penalty_transform
            )
            penalty_loader = DataLoader(
                penalty_set,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                generator=dataloader_generator(seed + 1),
                worker_init_fn=seed_worker,
            )
            penalty_loader_infinite = InfiniteDataloader(penalty_loader)
        else:
            penalty_loader_infinite = itertools.repeat(None)

        self.teacher_mean, self.teacher_std = teacher_normalization(self.teacher, train_loader)

        optimizer = torch.optim.Adam(
            itertools.chain(self.student.parameters(), self.autoencoder.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * self.train_steps), gamma=0.1
        )
        tqdm_obj = tqdm.tqdm(range(self.train_steps))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite
        ):
            image_st = image_st.to(device)
            image_ae = image_ae.to(device)
            if image_penalty is not None:
                image_penalty = image_penalty.to(device)
            with torch.no_grad():
                teacher_output_st = self.teacher(image_st)
                teacher_output_st = (teacher_output_st - self.teacher_mean) / self.teacher_std
            student_output_st = self.student(image_st)[:, : self.out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])

            if image_penalty is not None:
                student_output_penalty = self.student(image_penalty)[:, : self.out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard

            ae_output = self.autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = self.teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - self.teacher_mean) / self.teacher_std
            student_output_ae = self.student(image_ae)[:, self.out_channels :]
            distance_ae = (teacher_output_ae - ae_output) ** 2
            distance_stae = (ae_output - student_output_ae) ** 2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            if iteration % 10 == 0:
                tqdm_obj.set_description("Current loss: {:.4f}  ".format(loss_total.item()))

        self.teacher.eval()
        self.student.eval()
        self.autoencoder.eval()

        self.q_st_start, self.q_st_end, self.q_ae_start, self.q_ae_end = self.map_normalization(
            validation_loader=validation_loader
        )

    def map_normalization(self, validation_loader, desc="Intermediate map normalization"):
        maps_st, maps_ae = [], []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for image, _ in tqdm.tqdm(validation_loader, desc=desc):
            image = image.to(device)
            _, map_st, map_ae = self.predict(image=image)
            maps_st.append(map_st.to("cpu"))
            maps_ae.append(map_ae.to("cpu"))
            del image
            torch.cuda.empty_cache()
            gc.collect()
        maps_st = torch.cat(maps_st).to(device)
        maps_ae = torch.cat(maps_ae).to(device)
        return (
            torch.quantile(maps_st, q=0.9),
            torch.quantile(maps_st, q=0.995),
            torch.quantile(maps_ae, q=0.9),
            torch.quantile(maps_ae, q=0.995),
        )

    def predict(self, image):
        with torch.no_grad():
            teacher_output = self.teacher(image)
            teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
            student_output = self.student(image)
            autoencoder_output = self.autoencoder(image)
            map_st = torch.mean(
                (teacher_output - student_output[:, : self.out_channels]) ** 2,
                dim=1,
                keepdim=True,
            )
            map_ae = torch.mean(
                (autoencoder_output - student_output[:, self.out_channels :]) ** 2,
                dim=1,
                keepdim=True,
            )
            if self.q_st_start is not None:
                map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
            if self.q_ae_start is not None:
                map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
            map_combined = 0.5 * map_st + 0.5 * map_ae
        return map_combined, map_st, map_ae

    def predict_on_image(self, image_path):
        with torch.no_grad():
            image = PIL.Image.open(image_path).convert("RGB")
            image = self.default_transform(image)
            image = torch.unsqueeze(image, dim=0)
            image = image.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            map_combined, _, _ = self.predict(image=image)
            map_combined = transforms.CenterCrop((224, 224))(
                transforms.Resize((self.image_size, self.image_size))(map_combined)
            )
        return float(torch.max(map_combined).item()), map_combined[0, :, :, :].cpu().numpy()

    def reset(self):
        from methods.EffecientAd.common import get_autoencoder, get_pdn_medium, get_pdn_small

        if self.model_size == "small":
            self.teacher = get_pdn_small(self.out_channels)
            self.student = get_pdn_small(2 * self.out_channels)
        elif self.model_size == "medium":
            self.teacher = get_pdn_medium(self.out_channels)
            self.student = get_pdn_medium(2 * self.out_channels)
        else:
            raise ValueError(self.model_size)

        ensure_teacher_weights(self.weights, model_size=self.model_size)
        state_dict = sequential_teacher_state(torch.load(self.weights, map_location="cpu"))
        self.teacher.load_state_dict(state_dict)
        self.autoencoder = get_autoencoder(self.out_channels)

        self.teacher.eval()
        self.student.train()
        self.autoencoder.train()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.teacher.to(device)
        self.student.to(device)
        self.autoencoder.to(device)

        self.q_st_start = None
        self.q_st_end = None
        self.q_ae_start = None
        self.q_ae_end = None


effecientAD_Wrapper = EfficientADWrapper
