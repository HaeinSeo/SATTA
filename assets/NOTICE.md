# Figure attribution and scope

`satta_architecture.png` is the architecture diagram supplied with the SATTA
manuscript. Its transparent background has been composited onto white; labels,
arrows, photographs, and diagram geometry have not been redrawn.

The repository's MIT license covers the evaluation code and documentation.
It does not relicense the architecture artwork or embedded third-party
photographs and icons. Rights in those visual elements remain with their
respective owners. No dataset images, annotations, or model checkpoints are
distributed as part of the evaluation examples; the JSON examples are synthetic.

The diagram is schematic: the four size thresholds depend on predicted box
size, not semantic class. The boundary multiplier varies continuously with
distance; the exact formula and numerical defaults are in
[`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md).
