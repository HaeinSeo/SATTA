# Input and output formats

All coordinates are continuous original-image pixel coordinates. Boxes do not use an inclusive `+1` pixel convention. Class IDs must be nonnegative integers in the same dictionary for GT and predictions. Use `--num-classes N` to fix the class universe for aggregate legacy AP, particularly when a class has no GT in a subset.

## Ground truth

```json
{
  "image_a.png": {"width": 200, "height": 120, "boxes": [[10, 10, 20, 20, 0]]},
  "empty.png": {"width": 200, "height": 120, "boxes": []}
}
```

Each GT row is `[x0, y0, x1, y1, class_id]` with positive width and height. `boxes` is required. Optional `width` and `height` must be finite and positive; they document the coordinate frame and are not used to clip boxes. Include every evaluated image, including those with no objects. The evaluator does not infer an image split from prediction filenames.

This representation has no ignore/crowd fields. If native annotations have ignored regions or an `others` category, choose and document preprocessing before generating this GT. Dropping ignored GT alone does **not** exempt overlapping detections: they remain FP in this strict evaluation. An ignore-aware official evaluation is a separate protocol.

## Final predictions

```json
[
  {"image": "image_a.png", "bbox": [10, 10, 10, 10], "score": 0.9, "category_id": 0}
]
```

`bbox` is `[x, y, width, height]`. Scores must be finite and nonnegative; adjusted SATTA scores may exceed one. Negative widths/heights are rejected. Zero-area detections are retained as unmatched FP, not silently deleted. A GT image absent from this list receives zero predictions and can contribute FN. A prediction name absent from GT is an error. JSON list order is the stable tie order for equal scores within an image; GT mapping order determines cross-image equal-score AP ties.

## Optional candidate JSON

```json
{
  "meta": {"complete": true, "conf": 0.03},
  "preds": {
    "image_a.png": {
      "g": [[10, 10, 20, 20, 0.9, 0]],
      "t": [[10, 10, 20, 20, 0.8, 0]],
      "tl": [[0.5, 0.5]]
    },
    "empty.png": {"g": [], "t": [], "tl": []}
  }
}
```

`g` and `t` use `[x0, y0, x1, y1, score, class_id]`, unlike the final-detection `xywh` format. `g` holds full-image candidates; `t` holds tiled candidates after mapping to the original image. `tl[i]` holds the box center divided by tile width/height **in its originating tile**, in `[0,1]`, aligned with `t[i]`. Empty arrays are valid.

`meta.complete` must be `true`, `meta.conf` records the detector's actual candidate floor, and image keys must exactly match GT. A downstream threshold below that floor is rejected because discarded candidates cannot be recovered. Tiling, overlap, detector-internal NMS, resizing and full-image/tile inference belong to the producer and should be recorded in extra metadata. The CLI does not infer missing candidate provenance.

## Output

Use `--out outputs/result.json`. The CLI creates the parent directory and writes UTF-8 JSON. The `methods` mapping holds scalar counts and metrics, scale recalls, and optional `taxonomy`/`per_image` fields. `inputs` records SHA256 hashes; `source_sha256` identifies the evaluator source. `protocol` records the strict policy and metric conventions.

Undefined empty-band recall is `null`, while overall/XS ratios use the legacy zero-denominator convention documented in [METRICS.md](METRICS.md). Raw Python metric APIs retain their historical field names and NaN empty-band values; the CLI labels non-0.5 XS AP by the actual matching IoU and converts NaN to JSON null.
