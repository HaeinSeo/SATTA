# Metric definitions

This document defines the implemented evaluator. It separates the primary band-specific XS metrics from historical all-FP diagnostics and aggregate legacy AP. All methods in a comparison must use the same input images, GT preprocessing, confidence policy, and metric convention.

## Matching and scale

Detections are processed in descending score order, with stable input-order ties. Each takes the highest-IoU **unmatched GT of the same class** if IoU reaches the matching threshold (default 0.5). One GT and one detection can participate in at most one match. Unmatched detections are FP; unmatched GT are FN. Missing predictions for an annotated image do not remove that image from evaluation.

Object size is `sqrt((x1-x0)*(y1-y0))` in original-image pixels. Evaluation bands are **(lo, hi]**:

| Band | Interval in pixels |
|---|---|
| 1 | (0, 8] |
| 2 | (8, 12] |
| 3 | (12, 16] |
| 4 | (16, 20] |
| 5 | (20, 24] |
| 6 | (24, 32] |
| 7 | (32, 48] |
| 8 | (48, 64] |
| 9 | (64, 96] |
| 10 | (96, infinity) |

Groups are XS `(0,16]`, S `(16,32]`, M `(32,96]`, L `(96,infinity)`. These **evaluation boundaries differ intentionally from SATTA's predicted-size routing** at exactly 16, 32 and 96. A matched detection inherits the size of its GT; an unmatched detection is assigned by predicted size. Zero-area predictions remain FP and are assigned to XS.

## XS metrics

- `XS_recall = TP_XS / (TP_XS + FN_XS)`.
- `XS_precision = TP_XS / (TP_XS + FP_XS)`.
- `XS_AP50`: pool XS TP and XS FP over all images, sort by score, and integrate the all-point interpolated precision envelope at matching IoU 0.5. This is a pooled band AP, not class-macro AP or COCO AP_small.
- With CLI `--iou t`, the same scale AP is reported as `XS_AP_at_match_iou`. The `XS_AP50` alias appears only at `t=0.5`. The lower-level historical Python API keeps the `XS_AP50` field name at any `thr`; always read `match_iou`.

The all-point AP calculation constructs recall/precision from cumulative TP/FP counts, extends recall with 0 and 1 and precision with 1 and 0, takes the right-to-left maximum precision envelope, and sums `delta_recall * precision` at recall changes. GT size is used only by the evaluator, never by SATTA inference.

Secondary diagnostics `XS_precision_allFP` and `XS_AP50_allFP` charge **every unmatched detection of every predicted size** to XS. Matched non-XS TP do not enter their XS list. These are not interchangeable with the primary XS metrics. A large background FP lowers allFP XS precision but does not lower band-specific XS precision.

## Counts and denominators

- Recall in a band is matched GT / all GT in that band. Empty-band recall is undefined (`NaN` in Python, `null` in CLI JSON).
- `FP_per_img = FP / number_of_evaluated_images`.
- `FP_per_GT = FP / number_of_GT_objects`.
- Overall precision is `TP/(TP+FP)`; recall is `TP/(TP+FN)`.
- False discovery rate is `FP/(TP+FP)`. Detection here has no enumerated TN universe, so **classical FPR = FP/(FP+TN) is not reported**.
- FP reduction vs SAHI is `1 - FP_SATTA/FP_SAHI` on the exact same image set; undefined if SAHI FP is zero.
- Retained additional XS recall gain is `(R_SATTA-R_Global)/(R_SAHI-R_Global)` when the denominator is positive. It differs from the ratio `R_SATTA/R_SAHI`.

Legacy scalar ratios divide by `max(denominator,1)`: no predictions gives precision=0 and FDR=0; no XS GT gives XS recall/AP=0; with no GT anywhere, FP_per_GT is the raw FP count and should not be interpreted as a per-object rate. Inspect the reported counts before interpreting these edge cases. Per-image recall for an empty GT image is 1 in the legacy API, while aggregate recalls are count-based.

## Aggregate legacy AP

`legacy_AP50` and `legacy_AP50_95` preserve `satta.metrics.coco_ap` numerically. Despite that historical function name, it is **not official COCOeval or VisDrone AP**. It uses class-wise AP, a 101-point linearly interpolated precision envelope, trapezoidal integration, and no maxDets truncation. IoUs are 0.50:0.05:0.95. A single perfect detection evaluates to 0.995 under this convention, not 1.0.

By default the class universe is `0..max_GT_class_id`, including absent IDs as zero AP. Predictions with IDs above that inferred range still count as FP in count/XS metrics but do not enter legacy macro AP. Supply `--num-classes N` for the complete class dictionary; this averages zeros for absent classes and rejects labels outside the declared range.

The optional Python API `coco_evaluate` can call `pycocotools.COCOeval`, but it still receives this strict GT representation without native ignore/crowd metadata. It is a separate result, never a silent fallback for the legacy values and not a claim of official VisDrone equivalence.

## Ignore regions

This evaluator applies **strict matching** to the supplied GT. Detections in omitted ignore regions or on omitted `others` objects count as FP. There is no ignored-region suppression and no crowd matching. Report this policy alongside any result. To make a native VisDrone claim, evaluate the same detections with the official ignore-aware protocol separately and document the conversion, maximum detections, coordinates and category rules.
