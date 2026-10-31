# FP and FN taxonomy

Use `--taxonomy` to include `methods.<name>.taxonomy`, or call `satta.metrics.taxonomy(predictions, ground_truth, thr=0.5, bg_iou=0.1)`. Inputs follow the validated array layout in [INPUT_FORMAT.md](INPUT_FORMAT.md). The CLI validates inputs through the primary evaluator before taxonomy. This is a deterministic error categorization, not the TIDE software or TIDE's AP decomposition.

Start with the **same class-aware one-to-one matching** used for metrics. For each unmatched prediction, let `s` be its maximum IoU with any same-class GT and `o` its maximum IoU with any different-class GT. The ordered rules are:

| Priority | FP category | Rule |
|---|---|---|
| 1 | duplicate | `s >= matching_iou` |
| 2 | class_confusion | otherwise `o >= matching_iou` |
| 3 | localization | otherwise `s >= 0.1` |
| 4 | near_object | otherwise `o >= 0.1` |
| 5 | background | otherwise; also used when the image has no GT |

The precedence matters: categories are mutually exclusive even when several overlaps are nonzero. A duplicate overlaps a same-class GT sufficiently but lost the one-to-one matching competition.

For each unmatched GT:

| Priority | FN category | Rule |
|---|---|---|
| 1 | missed | no prediction, or maximum IoU over all predictions `<0.1` |
| 2 | localization | otherwise a same-class prediction has IoU `>=0.1` |
| 3 | class_confusion | otherwise only different-class predictions are nearby |

FN localization includes a same-class candidate that cannot be assigned because another GT won the match. Taxonomy searches all retained predictions and does not rematch or change TP/FP counts. FP category totals equal primary FP, and FN category totals equal primary FN. Tests cover these conservation identities.

The CLI requires a matching IoU at least 0.1 when taxonomy is enabled. The 0.1 boundary is a categorization setting, distinct from SATTA's NMS IoU and the evaluator's matching IoU. This taxonomy uses the strict ignore policy described in [METRICS.md](METRICS.md).
