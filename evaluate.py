#!/usr/bin/env python3
"""Evaluate final detections, or optionally merge global/tiled JSON candidates.

The core evaluator requires only NumPy. Optional SATTA/SAHI selection is loaded
only when --candidates is used. See docs/METRICS.md before interpreting AP.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from satta import __version__
from satta.io import json_safe, load_candidates, load_gt, load_predictions, read_json
from satta.metrics import coco_ap, evaluate_scale_resolved, taxonomy


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred", help="final detections JSON (xywh records); no additional filtering")
    source.add_argument("--candidates", help="safe global/tile candidates JSON; optional merge dependencies")
    parser.add_argument("--gt", required=True, help="ground-truth JSON image universe")
    parser.add_argument("--iou", type=float, default=.5, help="scale/taxonomy matching IoU; default .5")
    parser.add_argument("--methods", nargs="+", choices=["Global", "SAHI", "SATTA"],
                        default=["Global", "SAHI", "SATTA"], help="only for --candidates")
    parser.add_argument("--satta-cfg", help="SATTA JSON configuration; default is the shared preset")
    parser.add_argument("--conf", type=float, default=.03, help="Global/SAHI candidate threshold")
    parser.add_argument("--taxonomy", action="store_true", help="include the ordered FP/FN error taxonomy")
    parser.add_argument("--num-classes", type=int, help="class count for legacy macro AP (includes absent classes)")
    parser.add_argument("--per-image", action="store_true", help="include per-image counts")
    parser.add_argument("--out", type=Path, help="write a complete JSON report")
    return parser


def run(args):
    if not np.isfinite(args.iou) or not 0 < args.iou <= 1:
        raise ValueError("--iou must lie in (0, 1]")
    if not np.isfinite(args.conf) or args.conf < 0:
        raise ValueError("--conf must be finite and nonnegative")
    if args.num_classes is not None and args.num_classes <= 0:
        raise ValueError("--num-classes must be positive")
    if args.taxonomy and args.iou < .1:
        raise ValueError("--taxonomy requires --iou >= 0.1 (the background/localization boundary)")
    gts = load_gt(args.gt)
    config, candidate_meta = None, None
    if args.pred:
        if args.satta_cfg:
            raise ValueError("--satta-cfg applies only to --candidates")
        predictions = {"predictions": load_predictions(args.pred, gts)}
        input_path = args.pred
    else:
        from satta.satta import SattaConfig, satta_select
        from satta.baselines import global_only, sahi_merge
        cfg = SattaConfig(**read_json(args.satta_cfg)) if args.satta_cfg else SattaConfig()
        cfg.validate()
        required_floors = [args.conf] if any(m in args.methods for m in ("Global", "SAHI")) else []
        if "SATTA" in args.methods:
            required_floors.append(cfg.cand_conf if cfg.use_threshold else cfg.base_conf)
        floor = min(required_floors)
        candidate_meta, records = load_candidates(args.candidates, gts, floor)
        selectors = {"Global": lambda r: global_only(r, args.conf),
                     "SAHI": lambda r: sahi_merge(r, args.conf),
                     "SATTA": lambda r: satta_select(r, cfg)}
        predictions = {method: {name: selectors[method](record) for name, record in records.items()}
                       for method in dict.fromkeys(args.methods)}
        config, input_path = asdict(cfg), args.candidates
    methods = {}
    for method, pred in predictions.items():
        if args.num_classes is not None:
            for record in gts.values():
                if any(box[4] >= args.num_classes for box in record["boxes"]):
                    raise ValueError("GT category_id exceeds --num-classes")
            if any(len(d) and np.any(d[:, 5] >= args.num_classes) for d in pred.values()):
                raise ValueError("prediction category_id exceeds --num-classes")
        result = evaluate_scale_resolved(pred, gts, thr=args.iou)
        per_image = result.pop("_per_image")
        # The original AP50-named fields evaluate at args.iou; label exported
        # values by their actual meaning instead of misnaming a .3/.7 run.
        result["XS_AP_at_match_iou"] = result.pop("XS_AP50")
        result["XS_AP_at_match_iou_allFP"] = result.pop("XS_AP50_allFP")
        if args.iou == .5:
            result["XS_AP50"] = result["XS_AP_at_match_iou"]
            result["XS_AP50_allFP"] = result["XS_AP_at_match_iou_allFP"]
        result["legacy_AP50"], result["legacy_AP50_95"] = coco_ap(pred, gts, nc=args.num_classes)
        if args.taxonomy:
            fp, fn = taxonomy(pred, gts, thr=args.iou)
            result["taxonomy"] = {"FP": fp, "FN": fn, "background_iou": .1}
        if args.per_image:
            result["per_image"] = per_image
        methods[method] = result
    merge_packages = []
    if args.candidates and any(m in args.methods for m in ("SATTA", "SAHI")):
        merge_packages += ["torch", "torchvision"]
    if args.candidates and "SAHI" in args.methods:
        merge_packages.append("sahi")
    return json_safe({
        "toolkit_version": __version__,
        "protocol": {
            "matching": "score-descending stable greedy class-aware one-to-one",
            "scale_match_iou": args.iou, "XS": "GT sqrt(width*height) <= 16 px",
            "unmatched_scale_assignment": "predicted box size",
            "ignore_policy": "strict: no ignore regions or crowd/others exemptions",
            "aggregate_ap": "custom legacy 101-point interpolated trapezoidal AP; not official COCO/VisDrone",
            "coordinate_system": "original-image xyxy; continuous coordinates; no +1 pixel convention",
            "n_images": len(gts), "prediction_format": "candidates" if args.candidates else "detections",
            "global_sahi_conf": args.conf if args.candidates else None,
            "legacy_ap_num_classes": args.num_classes,
        },
        "inputs": {"ground_truth_sha256": sha256(args.gt), "predictions_sha256": sha256(input_path)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    **{name: importlib.metadata.version(name) for name in merge_packages}},
        "source_sha256": {str(path.relative_to(Path(__file__).parent)): sha256(path)
                          for path in [Path(__file__), *sorted((Path(__file__).parent / "satta").glob("*.py"))]},
        "satta_configuration": config,
        "candidate_metadata": candidate_meta,
        "methods": methods,
    })


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(f"{report['protocol']['n_images']} images | scale matching IoU={args.iou:g} | strict evaluation")
    print("method        XS recall   XS precision   XS AP@IoU   FP/img   legacy AP50")
    for name, m in report["methods"].items():
        print(f"{name:13s} {m['XS_recall']:9.4f} {m['XS_precision']:14.4f} "
              f"{m['XS_AP_at_match_iou']:11.4f} {m['FP_per_img']:8.2f} {m['legacy_AP50']:13.4f}")
        print(f"  secondary allFP XS precision={m['XS_precision_allFP']:.4f}; "
              f"allFP XS AP@IoU={m['XS_AP_at_match_iou_allFP']:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
