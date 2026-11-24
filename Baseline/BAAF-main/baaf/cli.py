"""Command-line interface for BAAF benchmarks."""
import argparse
import importlib
import json
import os
import sys
import traceback

from baaf.determinism import apply_runtime_env, environment_report

apply_runtime_env()


def _ensure_root_on_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_method_class(spec):
    """Return a wrapper class: built-in name, or ``module.path:ClassName``."""
    from wrappers import METHODS

    if spec in METHODS:
        module_name, class_name = METHODS[spec]
        return getattr(importlib.import_module(module_name), class_name)
    if ":" not in spec:
        raise ValueError(
            f"Unknown method {spec!r}. Use a built-in name ({', '.join(METHODS)}) "
            "or module.path:ClassName."
        )
    module_name, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def load_method(spec, faiss_on_gpu=False, imagenet_train_path=None):
    from wrappers import _EFFICIENTAD, _PATCHCORE

    cls = resolve_method_class(spec)
    if spec in _PATCHCORE:
        return cls(faiss_on_gpu=faiss_on_gpu)
    if spec in _EFFICIENTAD:
        kwargs = {}
        if imagenet_train_path:
            kwargs["imagenet_train_path"] = imagenet_train_path
        return cls(**kwargs)
    return cls()


def result_json_path(output_dir, method_name, seed, category, corruption, votes, bags):
    return os.path.join(
        output_dir,
        f"{method_name}_{seed}_{category}_num_neg_{str(corruption)}"
        f"_votes{str(votes)}_folds{str(bags)}_.json",
    )


def _add_run_args(parser):
    parser.add_argument("--data-dir", required=True, help="Dataset root (MVTec-style categories).")
    parser.add_argument("--output-dir", default="./results", help="Where to write JSON results.")
    parser.add_argument(
        "--method",
        default="patchcore",
        help="Built-in `patchcore` or `efficientad`, or module.path:ClassName.",
    )
    parser.add_argument("--votes", type=int, default=1, help="BAAF votes K. 0 = OCC baseline.")
    parser.add_argument("--bags", type=int, default=4, help="BAAF bags n.")
    parser.add_argument(
        "--corruption",
        type=float,
        default=0.1,
        help="Train anomaly rate. Values in [0, 1) are a fraction of |train|; "
        "values >= 1 are an absolute count.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0.")
    parser.add_argument(
        "--faiss-cpu",
        action="store_true",
        help="Run PatchCore FAISS on CPU (the default; kept for compatibility).",
    )
    parser.add_argument(
        "--faiss-gpu",
        action="store_true",
        help="Run PatchCore FAISS on GPU. Faster, not bit-reproducible across GPUs.",
    )
    parser.add_argument(
        "--imagenet-path",
        default=None,
        help="ImageNet/ImageNette folder for the EfficientAD penalty term. "
        "If omitted and the default path is missing, the penalty is disabled.",
    )
    parser.add_argument("--save-qualitative", action="store_true")
    parser.add_argument("--save-gmm-plots", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="Filter only; skip test evaluation.")
    parser.add_argument("--quiet", action="store_true")


def run_one(args, category, method=None):
    from baaf.algorithm import run_baaf

    cls = resolve_method_class(args.method)
    out = result_json_path(
        args.output_dir, cls.__name__, args.seed, category, args.corruption, args.votes, args.bags
    )
    if args.skip_existing and os.path.exists(out):
        print(f"Skipping existing {out}")
        return out
    if method is None:
        method = load_method(
            args.method,
            faiss_on_gpu=bool(getattr(args, "faiss_gpu", False))
            and not getattr(args, "faiss_cpu", False),
            imagenet_train_path=getattr(args, "imagenet_path", None),
        )
    print(f"Running {out}")
    _, path, results = run_baaf(
        category,
        method,
        data_dir=args.data_dir,
        seed=args.seed,
        k=args.bags,
        votes=args.votes,
        num_per_negative=args.corruption,
        output_dir=args.output_dir,
        save_qualitative_results=args.save_qualitative,
        evaluate_final_model=not args.no_eval,
        save_gmm_plots=args.save_gmm_plots,
        verbose=not args.quiet,
    )
    if results is not None and "img_results" in results:
        i_auroc = results["img_results"]["auroc"]
        p_auroc = results["pixel_results"]["auroc"]
        aupro = results["pixel_results"]["aupro_score"]
        print(
            f"{category}: I-AUROC={i_auroc:.4f}  P-AUROC={p_auroc:.4f}  AUPRO={aupro:.4f}"
        )
    return path


def cmd_run(args):
    if not args.category:
        raise SystemExit("--category is required for `baaf run`")
    run_one(args, args.category)


def cmd_benchmark(args):
    import gc

    import torch

    from baaf.constants import CATEGORIES

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.dataset:
        categories = CATEGORIES[args.dataset]
    else:
        raise SystemExit("Provide --dataset or --categories")

    for category in categories:
        try:
            run_one(args, category)
        except Exception:
            print(f"Error while running {category}")
            traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()


def cmd_prepare_visa(args):
    from baaf.data import convert_visa_to_mvtec_format

    convert_visa_to_mvtec_format(source=args.source, output=args.output)


def cmd_prepare_loco(args):
    from baaf.data import convert_loco_to_mvtec_format

    convert_loco_to_mvtec_format(source=args.source, output=args.output)


def cmd_summarize(args):
    from baaf.constants import CATEGORIES

    categories = args.categories
    if not categories and args.dataset:
        categories = CATEGORIES[args.dataset]
    if not categories:
        raise SystemExit("Provide --dataset or --categories")

    import glob
    import numpy as np

    files = list(glob.iglob(os.path.join(args.results_dir, "**", "*.json"), recursive=True))
    grouped = {}
    for path in files:
        name = os.path.basename(path)
        match = None
        for cat in sorted(categories, key=len, reverse=True):
            if f"_{cat}_" in name:
                match = cat
                break
        if match is None:
            continue
        with open(path) as f:
            data = json.load(f)
        key = name.replace(match, "{cat}")
        grouped.setdefault(key, []).append(
            (
                match,
                data["img_results"]["auroc"],
                data["pixel_results"]["auroc"],
                data["pixel_results"]["aupro_score"],
                data["hold_out_info"].get("final_hold_outs_precision"),
                data["hold_out_info"].get("final_hold_outs_recall"),
            )
        )

    for key, rows in grouped.items():
        found = [r[0] for r in rows]
        print(f"\n{key}")
        print(f"  categories ({len(found)}/{len(categories)}): {found}")
        print(f"  I-AUROC {np.mean([r[1] for r in rows]):.4f}")
        print(f"  P-AUROC {np.mean([r[2] for r in rows]):.4f}")
        print(f"  AUPRO   {np.mean([r[3] for r in rows]):.4f}")
        prec = [r[4] for r in rows if r[4] is not None]
        rec = [r[5] for r in rows if r[5] is not None]
        if prec:
            print(f"  filter precision {np.mean(prec):.4f}")
        if rec:
            print(f"  filter recall    {np.mean(rec):.4f}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="baaf",
        description="BAAF: wrap a one-class anomaly detector for unsupervised training.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run BAAF on a single category.")
    _add_run_args(run_p)
    run_p.add_argument("--category", required=True)

    bench_p = sub.add_parser("benchmark", help="Run BAAF on every category of a dataset.")
    _add_run_args(bench_p)
    bench_p.add_argument("--dataset", choices=["mvtec", "visa", "loco"])
    bench_p.add_argument("--categories", help="Comma-separated category list (overrides --dataset).")

    visa_p = sub.add_parser("prepare-visa", help="Convert VisA to MVTec folder layout.")
    visa_p.add_argument("--source", help="Path to the original VisA root.")
    visa_p.add_argument("--output", help="Output directory.")

    loco_p = sub.add_parser("prepare-loco", help="Convert MVTec LOCO AD to MVTec folder layout.")
    loco_p.add_argument("--source", help="Path to the original LOCO AD root.")
    loco_p.add_argument("--output", help="Output directory.")

    sum_p = sub.add_parser("summarize", help="Average JSON results over categories.")
    sum_p.add_argument("results_dir")
    sum_p.add_argument("--dataset", choices=["mvtec", "visa", "loco"])
    sum_p.add_argument("--categories", nargs="+")
    return parser


def main(argv=None):
    apply_runtime_env()
    _ensure_root_on_path()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "gpu", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    apply_runtime_env()
    if getattr(args, "command", None) in ("run", "benchmark") and not getattr(
        args, "quiet", False
    ):
        report = environment_report()
        print(
            "BAAF environment: python={python_version} torch={torch} "
            "cuda_built={torch_cuda_built} gpu={gpu_name} faiss={faiss}".format(
                python_version=report.get("python_version"),
                torch=report.get("torch"),
                torch_cuda_built=report.get("torch_cuda_built"),
                gpu_name=report.get("gpu_name"),
                faiss=report.get("faiss"),
            )
        )

    dispatch = {
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "prepare-visa": cmd_prepare_visa,
        "prepare-loco": cmd_prepare_loco,
        "summarize": cmd_summarize,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
