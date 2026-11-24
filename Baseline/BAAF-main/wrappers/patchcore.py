"""PatchCore ensemble wrapper. Hyperparameters match the paper configuration.

The underlying PatchCore library in ``methods/patchcore`` is the official
release (Roth et al., CVPR 2022) and is imported unmodified.
"""
import numpy as np
import PIL
import torch
from torchvision import transforms

from baaf.determinism import dataloader_generator, seed_worker
from wrappers.base import AD_Method_Wrapper


class PatchCoreWrapper(AD_Method_Wrapper):
    def __init__(self, faiss_on_gpu=False, faiss_num_workers=1):
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.layers_to_extract_from = (
            "0.layer2",
            "0.layer3",
            "1.layer2",
            "1.layer3",
            "2.features.denseblock2",
            "2.features.denseblock3",
        )
        self.backbone_names = ["wideresnet101", "resnext101", "densenet201"]
        self.pretrain_embed_dimension = 1024
        self.target_embed_dimension = 384
        self.patchsize = 3
        self.anomaly_scorer_num_nn = 1
        self.percentage = 0.01
        self.sampler_name = "approx_greedy_coreset"

        self._DATASETS = {
            "mvtec": ["methods.patchcore.datasets.mvtec", "MVTecDataset"]
        }
        self.name = "mvtec"
        self.train_val_split = 1.0
        self.batch_size = 2
        self.resize = (366, 366)
        self.imagesize = 320
        self.augment = False
        # Single-process loading: worker processes re-seed independently and
        # make PatchCore slightly OS/core-count dependent.
        self.num_workers = 0

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        self.reset()

    def train_on_class(self, seed, data_path):
        import methods.patchcore.utils
        from baaf.determinism import configure_determinism

        category = data_path.split("/")[-2]
        data_path = data_path.replace(category, "").replace("//", "/")

        configure_determinism(seed, warn_only=True)
        dataloaders = self.get_dataloaders(data_path, category, seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        methods.patchcore.utils.fix_seeds(seed, device)

        with torch.no_grad():
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = self.get_sampler(device)
            self.PatchCore_list = self.get_patchcore(imagesize, sampler, device)

            for PatchCore in self.PatchCore_list:
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    methods.patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

    def get_dataloaders(self, data_path, subdataset, seed):
        dataset_info = self._DATASETS[self.name]
        dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

        train_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=self.resize,
            train_val_split=self.train_val_split,
            imagesize=self.imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=seed,
            augment=self.augment,
        )
        test_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=self.resize,
            imagesize=self.imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=seed,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker if self.num_workers else None,
            generator=dataloader_generator(seed),
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker if self.num_workers else None,
            generator=dataloader_generator(seed),
        )
        train_dataloader.name = self.name
        if subdataset is not None:
            train_dataloader.name += "_" + subdataset
        return {
            "training": train_dataloader,
            "validation": None,
            "testing": test_dataloader,
        }

    def predict_on_image(self, image_path):
        with torch.no_grad():
            image = PIL.Image.open(image_path).convert("RGB")
            image = self.transform_img(image)
            image = torch.unsqueeze(image, dim=0)
            image = image.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            scores, masks = [], []
            for PatchCore in self.PatchCore_list:
                image_score, mask = PatchCore.predict(image)
                scores.append(image_score)
                masks.append(mask)
            score_final = np.mean(scores).item()
            mask_final = np.mean(np.array(masks), axis=0)
            mask_final = (
                transforms.Resize((224, 224))(torch.from_numpy(mask_final))
                .cpu()
                .numpy()
            )
        return score_final, mask_final

    def reset(self):
        if len(self.backbone_names) > 1:
            self.layers_to_extract_from_coll = [
                [] for _ in range(len(self.backbone_names))
            ]
            for layer in self.layers_to_extract_from:
                idx = int(layer.split(".")[0])
                layer = ".".join(layer.split(".")[1:])
                self.layers_to_extract_from_coll[idx].append(layer)
        else:
            self.layers_to_extract_from_coll = [self.layers_to_extract_from]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sampler = self.get_sampler(device)
        self.PatchCore_list = self.get_patchcore(
            (3, self.imagesize, self.imagesize), sampler, device
        )

    def get_sampler(self, device):
        import methods.patchcore.sampler

        if self.sampler_name == "identity":
            return methods.patchcore.sampler.IdentitySampler()
        if self.sampler_name == "greedy_coreset":
            return methods.patchcore.sampler.GreedyCoresetSampler(
                self.percentage, device
            )
        if self.sampler_name == "approx_greedy_coreset":
            return methods.patchcore.sampler.ApproximateGreedyCoresetSampler(
                self.percentage, device
            )
        raise ValueError("Invalid sampler name chosen!")

    def get_patchcore(self, input_shape, sampler, device):
        import methods.patchcore.backbones
        import methods.patchcore.common
        import methods.patchcore.patchcore

        loaded = []
        for backbone_name, layers_to_extract_from in zip(
            self.backbone_names, self.layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = (
                    backbone_name.split(".seed-")[0],
                    int(backbone_name.split("-")[-1]),
                )
            backbone = methods.patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed
            nn_method = methods.patchcore.common.FaissNN(
                self.faiss_on_gpu, self.faiss_num_workers
            )
            instance = methods.patchcore.patchcore.PatchCore(device)
            instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=self.pretrain_embed_dimension,
                target_embed_dimension=self.target_embed_dimension,
                patchsize=self.patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=self.anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded.append(instance)
        return loaded


# Original class name, kept so result JSON filenames stay recognisable.
patchcore_Wrapper = PatchCoreWrapper
