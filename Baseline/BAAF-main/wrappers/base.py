"""Minimal interface every OCC method must implement to be wrapped by BAAF.

BAAF never inspects model internals. It only changes which images are present
in the training folder passed to ``train_on_class``.
"""


class AD_Method_Wrapper:
    """One-class classifier adapter.

    Implement these three methods and BAAF can wrap the detector without any
    other changes to training or inference:

    * ``reset()`` — construct a fresh, untrained model (called before every fit).
    * ``train_on_class(seed, data_path)`` — fit on ``data_path/train/good/*.png``.
      ``data_path`` is a category folder with a trailing slash, e.g.
      ``./datasets/MVTecAD/bottle/``.
    * ``predict_on_image(image_path)`` — return ``(image_score, anomaly_map)``
      where ``image_score`` is a float (higher = more anomalous) and
      ``anomaly_map`` is a numpy array of shape ``(1, H, W)``.
    """

    def train_on_class(self, seed, data_path):
        raise NotImplementedError

    def predict_on_image(self, image_path):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
