from wrappers.base import AD_Method_Wrapper  # re-export

METHODS = {
    "patchcore": ("wrappers.patchcore", "PatchCoreWrapper"),
    "patchcore_Wrapper": ("wrappers.patchcore", "PatchCoreWrapper"),
    "efficientad": ("wrappers.efficientad", "EfficientADWrapper"),
    "efficientad_Wrapper": ("wrappers.efficientad", "EfficientADWrapper"),
    "effecientAD_Wrapper": ("wrappers.efficientad", "EfficientADWrapper"),
}

_PATCHCORE = {"patchcore", "patchcore_Wrapper"}
_EFFICIENTAD = {"efficientad", "efficientad_Wrapper", "effecientAD_Wrapper"}


def get_method(name, **kwargs):
    if name not in METHODS:
        known = ", ".join(sorted(set(METHODS)))
        raise ValueError(f"Unknown method {name!r}. Built-in: {known}")
    module_name, class_name = METHODS[name]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)
