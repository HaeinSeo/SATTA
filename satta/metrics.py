"""Scale-resolved evaluation protocol (paper Sec. 4.3) + FP/FN taxonomy + object transitions."""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x renamed trapz to trapezoid

BANDS = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 32), (32, 48), (48, 64), (64, 96), (96, np.inf)]
XS_MAX = 16.0
GROUPS = {"XS": (0, 16), "S": (16, 32), "M": (32, 96), "L": (96, np.inf)}
EMPTY = np.zeros((0, 6), np.float32)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def size_of(b):
    return np.sqrt(np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0))


def gt_array(g):
    a = np.asarray(g["boxes"], np.float32).reshape(-1, 5)
    _validate_boxes(a, 4, "ground truth")
    return a


def _validate_boxes(a, class_col, label, allow_zero=False):
    if not np.isfinite(a).all():
        raise ValueError(f"{label} contains non-finite values")
    invalid_geometry = a[:, 2:4] < a[:, :2] if allow_zero else a[:, 2:4] <= a[:, :2]
    if len(a) and np.any(invalid_geometry):
        raise ValueError(f"{label} contains boxes with {'negative' if allow_zero else 'non-positive'} width or height")
    classes = a[:, class_col]
    if np.any(classes < 0) or np.any(classes != np.floor(classes)):
        raise ValueError(f"{label} class IDs must be nonnegative integers")


def _prediction_array(det):
    a = np.asarray(det, dtype=np.float32)
    if a.size == 0:
        return EMPTY
    if a.ndim != 2 or a.shape[1] != 6:
        raise ValueError("predictions must be an Nx6 array (xyxy, score, class)")
    # Clipping can leave zero-area predictions. Keep them as unmatched FP;
    # dropping them would artificially improve the detector's false-positive rate.
    _validate_boxes(a, 5, "predictions", allow_zero=True)
    if np.any(a[:, 4] < 0):
        raise ValueError("prediction scores must be nonnegative")
    return a


def _validate_eval_inputs(preds, gts):
    if not gts:
        raise ValueError("evaluation requires at least one annotated image")
    unknown = set(preds) - set(gts)
    if unknown:
        raise ValueError(f"predictions contain {len(unknown)} image(s) absent from ground truth: "
                         f"{sorted(unknown)[:3]}")


def _validate_iou_threshold(thr):
    if not np.isfinite(thr) or not 0 < thr <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")


def match_image(det, gt, thr=0.5):
    """One-to-one greedy matching: descending confidence, class-aware, best-IoU unmatched GT >= thr.

    Returns det->gt index (-1 if unmatched) and the full IoU matrix.
    """
    _validate_iou_threshold(thr)
    iou = box_iou(det[:, :4], gt[:, :4])
    m = -np.ones(len(det), int)
    if not len(gt) or not len(det):
        return m, iou
    taken = np.zeros(len(gt), bool)
    same = det[:, 5:6] == gt[None, :, 4]
    for i in np.argsort(-det[:, 4], kind="stable"):
        c = np.where(same[i] & ~taken, iou[i], -1)
        j = int(c.argmax())
        if c[j] >= thr:
            m[i] = j
            taken[j] = True
    return m, iou


def _ap(tp, n_gt):
    """All-point interpolated AP from a score-sorted TP-flag vector (1=TP, 0=FP)."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def evaluate_scale_resolved(preds, gts, thr=0.5):
    """Evaluate a fixed image universe, counting missing predictions as no detections.

    preds maps image name to an Nx6 array (xyxy, conf, cls); gts maps it to
    a dict with Nx5 boxes. Unknown prediction images and an empty image universe
    are errors rather than silently discarded evidence. FP/image is not FPR:
    detection has no enumerated true-negative universe. false_discovery_rate
    means FP/(TP+FP), i.e. one minus overall precision when predictions exist,
    at this operating point. Precision and FDR are both 0 for no predictions.
    AP50-named legacy fields evaluate at ``thr``; ``match_iou`` records it.
    """
    _validate_eval_inputs(preds, gts)
    _validate_iou_threshold(thr)
    edges = [b[1] for b in BANDS[:-1]]
    rec_band, n_band = np.zeros(len(BANDS)), np.zeros(len(BANDS))
    tp_xs = fn_xs = fp_xs = n_fp = n_tp = n_fn = 0
    xs_scores, xs_flags, per_image = [], [], {}
    pa_scores, pa_flags = [], []   # paper-style XS list: XS true positives + every unmatched detection
    for name, g in gts.items():
        gt = gt_array(g)
        det = _prediction_array(preds.get(name, EMPTY))
        m, _ = match_image(det, gt, thr)
        gs = size_of(gt) if len(gt) else np.zeros(0)
        gband = np.digitize(gs, edges, right=True)             # band (lo, hi]
        hit = np.zeros(len(gt), bool)
        hit[m[m >= 0]] = True
        for b in range(len(BANDS)):
            sel = gband == b
            n_band[b] += sel.sum()
            rec_band[b] += hit[sel].sum()
        xs_gt = gs <= XS_MAX
        tp_xs += int((hit & xs_gt).sum())
        fn_xs += int((~hit & xs_gt).sum())
        matched = m >= 0
        # matched detections inherit the band of their GT, unmatched ones use the predicted box size
        det_xs = size_of(det[:, :4]) <= XS_MAX if len(det) else np.zeros(0, bool)
        if len(gt) and matched.any():
            det_xs[matched] = xs_gt[m[matched]]
        fp_xs += int((~matched & det_xs).sum())
        keep_pa = ~matched | (det_xs if len(det) else matched)
        pa_scores.append(det[keep_pa, 4])
        pa_flags.append(matched[keep_pa].astype(np.int8))
        xs_scores.append(det[det_xs, 4])
        xs_flags.append(matched[det_xs].astype(np.int8))
        n_tp += int(matched.sum())
        n_fp += int((~matched).sum())
        n_fn += int((~hit).sum())
        per_image[name] = dict(tp=int(matched.sum()), fp=int((~matched).sum()), gt=len(gt), npred=len(det),
                               tp_xs=int((hit & xs_gt).sum()), n_xs=int(xs_gt.sum()),
                               fn_xs=int((~hit & xs_gt).sum()),
                               recall=float(hit.mean()) if len(gt) else 1.0)
    sc, fl = np.concatenate(xs_scores), np.concatenate(xs_flags)
    o = np.argsort(-sc, kind="stable")
    psc, pfl = np.concatenate(pa_scores), np.concatenate(pa_flags)
    po = np.argsort(-psc, kind="stable")
    n_img, n_gt = len(gts), int(n_band.sum())
    out = dict(
        XS_recall=tp_xs / max(tp_xs + fn_xs, 1),
        XS_precision=tp_xs / max(tp_xs + fp_xs, 1),
        XS_AP50=_ap(fl[o], tp_xs + fn_xs),
        XS_precision_allFP=tp_xs / max(tp_xs + n_fp, 1),
        XS_AP50_allFP=_ap(pfl[po], tp_xs + fn_xs),
        FP_per_img=n_fp / n_img, FP_per_GT=n_fp / max(n_gt, 1),
        precision=n_tp / max(n_tp + n_fp, 1), recall=n_tp / max(n_gt, 1),
        false_discovery_rate=n_fp / max(n_tp + n_fp, 1),
        TP_XS=tp_xs, FN_XS=fn_xs, FP_XS=fp_xs,
        n_images=n_img, n_predictions=n_tp + n_fp, match_iou=float(thr),
        TP=n_tp, FP=n_fp, FN=n_fn, n_gt=n_gt, n_gt_xs=tp_xs + fn_xs,
        recall_by_band={f"{lo}-{hi}": float(rec_band[i] / n_band[i]) if n_band[i] else float("nan")
                        for i, (lo, hi) in enumerate(BANDS)},
        n_by_band={f"{lo}-{hi}": int(n_band[i]) for i, (lo, hi) in enumerate(BANDS)},
    )
    out["recall_by_group"] = {k: _group_recall(rec_band, n_band, lo, hi) for k, (lo, hi) in GROUPS.items()}
    out["_per_image"] = per_image
    return out


def _group_recall(rec_band, n_band, lo, hi):
    idx = [i for i, (a, b) in enumerate(BANDS) if a >= lo and b <= hi]
    n = n_band[idx].sum()
    return float(rec_band[idx].sum() / n) if n else float("nan")


# ------------------------------------------------------------ aggregate AP (Ultralytics-style)
def coco_ap(preds, gts, nc=None):
    """Legacy repo AP50/AP50-95, kept numerically compatible with old reports.

    This uses trapezoidal integration of linearly interpolated precision at 101
    recall points (Ultralytics-style), no maxDets cap, and no crowd/ignore
    handling. It is NOT official COCOeval. With ``nc`` it also averages zeros
    for absent classes. Use ``coco_evaluate`` for a separately labelled COCOeval
    result; never mix these two conventions in one comparison.
    """
    _validate_eval_inputs(preds, gts)
    nc = nc or int(max(gt_array(g)[:, 4].max(initial=0) for g in gts.values())) + 1
    ths = np.linspace(0.5, 0.95, 10)
    recs = {c: [] for c in range(nc)}
    ngt = np.zeros(nc)
    for name, g in gts.items():
        gt = gt_array(g)
        det = _prediction_array(preds.get(name, EMPTY))
        for c in gt[:, 4].astype(int):
            ngt[c] += 1
        flags = np.zeros((len(det), len(ths)), bool)
        for ti, t in enumerate(ths):
            flags[:, ti] = match_image(det, gt, t)[0] >= 0
        for c in range(nc):
            s = det[:, 5] == c
            if s.any():
                recs[c].append((det[s, 4], flags[s]))
    aps = np.zeros((nc, len(ths)))
    x = np.linspace(0, 1, 101)
    for c in range(nc):
        if not recs[c] or ngt[c] == 0:
            continue
        sc = np.concatenate([r[0] for r in recs[c]])
        fl = np.concatenate([r[1] for r in recs[c]])[np.argsort(-sc, kind="stable")]
        for ti in range(len(ths)):
            ctp, cfp = np.cumsum(fl[:, ti]), np.cumsum(~fl[:, ti])
            r, p = ctp / ngt[c], ctp / np.maximum(ctp + cfp, 1e-9)
            mrec, mpre = np.concatenate([[0], r, [1]]), np.concatenate([[1], p, [0]])
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            aps[c, ti] = _trapz(np.interp(x, mrec, mpre), x)
    return float(aps[:, 0].mean()), float(aps.mean())


def coco_evaluate(preds, gts, max_dets=100):
    """Run actual pycocotools COCOeval on this repo's *strict* GT representation.

    Defaults to maxDets=[1, 10, 100], IoU=.50:.05:.95 and 101 recall points.
    The input has no crowd/ignore annotations, so this does not reproduce a
    dataset's native ignore-aware protocol (e.g. VisDrone). Confidence filtering
    has already happened upstream. Undefined statistics are returned as None.
    pycocotools is an optional dependency; its absence is never silently replaced
    with the legacy metric.
    """
    import contextlib
    import io

    _validate_eval_inputs(preds, gts)
    if not isinstance(max_dets, (int, np.integer)) or max_dets < 10:
        raise ValueError("max_dets must be an integer >= 10")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("coco_evaluate requires pycocotools; install it with pip install pycocotools") from exc

    images, annotations, detections, categories = [], [], [], set()
    for image_id, name in enumerate(sorted(gts), 1):
        g = gts[name]
        images.append({"id": image_id, "file_name": name,
                       "width": g.get("width", 0), "height": g.get("height", 0)})
        for x0, y0, x1, y1, cls in gt_array(g):
            cid = int(cls)
            categories.add(cid)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": cid, "bbox": [float(x0), float(y0), float(x1-x0), float(y1-y0)],
                                "area": float((x1-x0) * (y1-y0)), "iscrowd": 0})
        for x0, y0, x1, y1, score, cls in _prediction_array(preds.get(name, EMPTY)):
            cid = int(cls)
            categories.add(cid)
            detections.append({"image_id": image_id, "category_id": cid,
                               "bbox": [float(x0), float(y0), float(x1-x0), float(y1-y0)],
                               "score": float(score)})
    dataset = {"images": images, "annotations": annotations,
               "categories": [{"id": c, "name": str(c)} for c in sorted(categories)], "info": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_api = COCO()
        gt_api.dataset = dataset
        gt_api.createIndex()
        if detections:
            det_api = gt_api.loadRes(detections)
        else:
            # COCO.loadRes indexes annotations[0] and fails for empty predictions.
            det_api = COCO()
            det_api.dataset = {**dataset, "annotations": []}
            det_api.createIndex()
        evaluator = COCOeval(gt_api, det_api, "bbox")
        evaluator.params.maxDets = [1, 10, int(max_dets)]
        evaluator.evaluate()
        evaluator.accumulate()
        # Read tensors directly: COCOeval.summarize() hard-codes maxDets=100
        # for its overall AP even when the configured maximum is different.
        precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDet
        recall = evaluator.eval["recall"]
        def mean_valid(a):
            a = a[a > -1]
            return float(a.mean()) if a.size else None
        result = {"AP50_95": mean_valid(precision[:, :, :, 0, -1]),
                  "AP50": mean_valid(precision[0, :, :, 0, -1]),
                  "AP75": mean_valid(precision[5, :, :, 0, -1]),
                  "AP_small": mean_valid(precision[:, :, :, 1, -1]),
                  "AP_medium": mean_valid(precision[:, :, :, 2, -1]),
                  "AP_large": mean_valid(precision[:, :, :, 3, -1]),
                  "AR_max": mean_valid(recall[:, :, 0, -1])}
    result["protocol"] = {"evaluator": "pycocotools.COCOeval", "iou_type": "bbox",
                          "max_dets": [1, 10, int(max_dets)], "ignore_policy": "strict_no_ignore_annotations",
                          "n_images": len(gts), "n_gt": len(annotations), "n_predictions": len(detections)}
    return result


# ------------------------------------------------------------ taxonomy and object transitions
FP_TYPES = ["background", "localization", "class_confusion", "near_object", "duplicate"]
FN_TYPES = ["missed", "localization", "class_confusion"]


def taxonomy(preds, gts, thr=0.5, bg_iou=0.1):
    """TIDE-style split. FP: duplicate > class confusion > localization > near-object > background."""
    fp, fn = dict.fromkeys(FP_TYPES, 0), dict.fromkeys(FN_TYPES, 0)
    for name, g in gts.items():
        gt = gt_array(g)
        det = preds.get(name, EMPTY)
        m, iou = match_image(det, gt, thr)
        same = (det[:, 5:6] == gt[None, :, 4]) if len(gt) else np.zeros((len(det), 0), bool)
        for i in np.where(m < 0)[0]:
            if not len(gt):
                fp["background"] += 1
                continue
            s, o = np.where(same[i], iou[i], 0).max(), np.where(~same[i], iou[i], 0).max()
            if s >= thr:
                fp["duplicate"] += 1
            elif o >= thr:
                fp["class_confusion"] += 1
            elif s >= bg_iou:
                fp["localization"] += 1
            elif o >= bg_iou:
                fp["near_object"] += 1
            else:
                fp["background"] += 1
        hit = np.zeros(len(gt), bool)
        hit[m[m >= 0]] = True
        for j in np.where(~hit)[0]:
            if not len(det) or iou[:, j].max() < bg_iou:
                fn["missed"] += 1
                continue
            col, sm = iou[:, j], same[:, j]
            if (col * sm).max() >= bg_iou:
                fn["localization"] += 1
            else:
                fn["class_confusion"] += 1
    fp["total"], fn["total"] = sum(fp.values()), sum(fn.values())
    return fp, fn


def transitions(pred_a, pred_b, gts, thr=0.5):
    """Object-level A to B: recovered TP (FN in A, TP in B), lost TP, kept TP, persistent FN."""
    t = dict(recovered_TP=0, lost_TP=0, kept_TP=0, persistent_FN=0, recovered_xs=0, persistent_xs=0, n_xs=0)
    for name, g in gts.items():
        gt = gt_array(g)
        if not len(gt):
            continue
        hits = []
        for p in (pred_a, pred_b):
            m, _ = match_image(p.get(name, EMPTY), gt, thr)
            h = np.zeros(len(gt), bool)
            h[m[m >= 0]] = True
            hits.append(h)
        a, b = hits
        xs = size_of(gt) <= XS_MAX
        t["recovered_TP"] += int((~a & b).sum())
        t["lost_TP"] += int((a & ~b).sum())
        t["kept_TP"] += int((a & b).sum())
        t["persistent_FN"] += int((~a & ~b).sum())
        t["recovered_xs"] += int((~a & b & xs).sum())
        t["persistent_xs"] += int((~a & ~b & xs).sum())
        t["n_xs"] += int(xs.sum())
    return t
