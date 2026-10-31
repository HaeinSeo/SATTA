# Shared SATTA configuration

The complete executable preset is [configs/default.json](../configs/default.json), matching the dataclass in `satta/satta.py`. SATTA receives global and tiled candidates in original-image coordinates and uses **predicted boxes only** for routing. It does not access GT sizes or labels during selection.

| Setting | Value |
|---|---|
| Predicted-size XS / S / M / L buckets | `<16`, `[16,32)`, `[32,96)`, `>=96` px |
| Size | `sqrt(predicted_width * predicted_height)` |
| XS/S tile score multiplier | 1.10 |
| M/L global score multiplier | 1.10 |
| M/L tile score multiplier | 0.85 |
| All other source/size multiplier | 1.00 |
| Boundary minimum multiplier gamma | 0.65 |
| Boundary margin m | 0.08 |
| XS / S adjusted-score threshold | 0.055 / 0.055 |
| M / L adjusted-score threshold | 0.10 / 0.12 |
| Raw candidate confidence floor | 0.055 |
| Final class-wise NMS IoU | 0.50 |
| Component switches | source/size, boundary, thresholds all enabled |
| Uniform ablation threshold (`base_conf`) | 0.03, used only when thresholding component is disabled |

## Selection order

1. Concatenate full-image and tile detections; filter by the raw candidate confidence floor.
2. Compute predicted size and route to XS/S/M/L.
3. Multiply confidence by the source/size factor above.
4. For a tile candidate with normalized in-tile center `(u,v)`, set `delta=min(u,1-u,v,1-v)` and multiply by `beta=gamma+(1-gamma)*min(delta/m,1)`. Global candidates are not boundary-penalized.
5. Keep candidates whose adjusted score reaches their size-dependent threshold.
6. Apply class-wise NMS at IoU 0.5. Returned scores are the adjusted scores.

The boundary penalty is continuous, not a hard edge/non-edge switch. Exactly 16px routes to S during inference but belongs to XS during evaluation; the two boundary conventions serve different purposes and are explicit.

## Upstream inference settings

For the shared study preset, detector candidate generation used input confidence 0.03 for the common Global/SAHI candidate cache, detector IoU 0.60, overlap 0.25, tile sizes 768px for the 960px configuration and 640px for the 640px configurations. SATTA then applies its raw 0.055 floor. These producer settings are **not implemented by this minimal selection/evaluation package** and must be recorded by the prediction producer. Global plus tile candidates are both used for SAHI.

The optional selector preserves the original Torchvision NMS calculation; ties can depend on the installed NMS implementation/device. The included examples run on CPU. Learned detector weights, inference resizing, internal maxDets and pre-NMS filtering also affect the candidate population and cannot be reconstructed from configuration values alone.

## Configuration variants

To study a new policy, create a separate JSON file and pass `--satta-cfg path.json`. The resulting report includes every selected value. Select parameters on validation data and freeze them before test evaluation. A per-detector calibrated policy is a different experiment from transferring this common preset across detectors. Disabling a component exposes its effect but does not prove an independent causal contribution when components interact.
