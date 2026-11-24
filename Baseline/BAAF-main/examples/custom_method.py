"""Minimal OCC wrapper used as a template for custom methods.

This is not a competitive detector. It exists so you can:

1. See the three methods BAAF requires.
2. Run the full BAAF pipeline on CPU without PatchCore.

Usage::

    python -m baaf run \\
        --method examples.custom_method:PixelStatsWrapper \\
        --data-dir ./datasets/MVTecAD \\
        --category toothbrush \\
        --votes 1 --bags 4 --corruption 0.1
"""
import glob
import os

import numpy as np
from PIL import Image

from wrappers.base import AD_Method_Wrapper


class PixelStatsWrapper(AD_Method_Wrapper):
    """Nearest-neighbour on a 32x32 greyscale thumbnail of each training image."""

    def __init__(self):
        self.size = 32
        self.memory = None

    def reset(self):
        self.memory = None

    def _embed(self, image_path):
        image = Image.open(image_path).convert("L").resize((self.size, self.size))
        return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0

    def train_on_class(self, seed, data_path):
        paths = sorted(glob.glob(os.path.join(data_path, "train", "good", "*.png")))
        if not paths:
            raise FileNotFoundError(f"No PNGs in {data_path}train/good/")
        self.memory = np.stack([self._embed(p) for p in paths], axis=0)

    def predict_on_image(self, image_path):
        query = self._embed(image_path)
        distances = np.linalg.norm(self.memory - query[None, :], axis=1)
        score = float(distances.min())
        # Constant dummy map; BAAF filtering uses only the image-level score.
        h = w = 224
        return score, np.full((1, h, w), score, dtype=np.float32)
