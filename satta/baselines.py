"""Global inference and SAHI baselines built on the *official* SAHI postprocess code."""
import numpy as np


def global_only(rec, conf=0.03):
    d = rec["g"]
    return d[d[:, 4] >= conf]


def sahi_merge(rec, conf=0.03, postprocess="NMS", match_metric="IOU", match_threshold=0.5,
               class_agnostic=False, perform_standard_pred=True):
    """SAHI = full-image prediction + tile predictions merged by SAHI's postprocess.

    postprocess: NMS | GREEDYNMM | NMM | LSNMS (SAHI names; get_sliced_prediction auto-switches to NMS+IOU when conf < LOW_MODEL_CONFIDENCE) or 'TORCH_NMS' (fast class-wise NMS).
    """
    parts = [rec["t"]]
    if perform_standard_pred:
        parts.append(rec["g"])
    d = np.concatenate(parts, 0)
    d = d[d[:, 4] >= conf]
    if len(d) == 0:
        return d[:, :6]
    d = d[:, :6]
    if postprocess == "TORCH_NMS":
        import torch
        from torchvision.ops import batched_nms
        d = d.copy()
        d[:, :4] = np.maximum(d[:, :4], 0)    # the SAHI library clips negative coordinates to 0 (RT-DETR boxes may stick out of the image); mirror it
        keep = batched_nms(torch.from_numpy(d[:, :4]).float(), torch.from_numpy(d[:, 4]).float(),
                           torch.from_numpy(d[:, 5]).long(), match_threshold).numpy()
        return d[keep]
    from sahi.predict import POSTPROCESS_NAME_TO_CLASS
    from sahi.prediction import ObjectPrediction
    pp = POSTPROCESS_NAME_TO_CLASS[postprocess](
        match_threshold=match_threshold, match_metric=match_metric, class_agnostic=class_agnostic)
    preds = [ObjectPrediction(bbox=[float(r[0]), float(r[1]), float(r[2]), float(r[3])],
                              category_id=int(r[5]), category_name=str(int(r[5])), score=float(r[4]))
             for r in d]
    out = pp(preds)
    return np.array([[*o.bbox.to_xyxy(), o.score.value, o.category.id] for o in out], dtype=np.float32).reshape(-1, 6)
