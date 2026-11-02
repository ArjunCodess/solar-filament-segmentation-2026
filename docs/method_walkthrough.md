# How this solution works

The algorithm predicts filament pixels first, then groups them into distinct
filaments. Its main unresolved errors are missing thin extensions, including
background, and assigning the wrong instance identity to nearby fragments.
The notebook's PQ is a local reference, and the organizer has confirmed that
hidden scoring contains additional checks.

## Images and supervision

The competition release contains 707 physical training observations but 1,154
annotation views. Two annotators can disagree about a filament without either
view being a separate physical observation. We retain every approved view as a
training target and keep all views of one image together in validation. A three-day
connected temporal grouping prevents consecutive-day images crossing folds.
It does not prove independence of all long-lived filaments.

Only approved polygon/RLE masks supervise the models. The skeleton and boundary
targets are calculated from those masks. Supplied principal-spine metadata is
not used, because the rules prohibit other ground-truth metadata. No public
full-release test labels, test pseudo-label targets, or adjacent test frames
are imported.

## Illumination correction

The disk detector fits the limb to estimate its center and radius. A radial
median profile estimates the brightness change from center to limb. Dividing
by this profile reduces limb darkening, then CLAHE improves local contrast.
The original grayscale image is retained as a second channel because correction
can suppress useful signals or amplify noise. Invalid disk geometry causes an
explicit failure; it does not silently become a guessed circle.

## Model and training

SegFormer-B3 and ConvNeXt-Small U-Net++ are alternative pretrained image encoders
and decoders. A separate full-resolution branch retains detail before the
encoder downsamples it. A 128-pixel full-disk context image supplies global
appearance, but its pooled representation does not explicitly locate a crop
on the solar disk. The output channels are mask probability, mask-derived
skeleton heatmap, and instance boundary probability.

The original foreground loss combines Dice, focal, and clDice. Dice measures
overlap, focal emphasizes difficult pixels, and clDice compares skeleton
coverage and foreground topology. Two auxiliary binary cross-entropy terms
supervise the skeleton and boundary heads. These terms can conflict with
instance extraction, so the new paired experiment compares them with a
mask-only Dice/focal continuation under the same augmentation and training budget.

The multiscale continuation draws 480-800 native-pixel regions and resizes them
to 640 pixels. Targets use nearest-neighbor resizing to preserve categorical
mask values. A source region larger than 640 sacrifices some native detail,
while a smaller region magnifies it; this is an experiment to measure rather
than an assumed improvement. Both new variants sample physical images uniformly
before choosing one annotation view, avoiding extra sampling weight for images
with more annotators.

## Instances and inference

Overlapping image tiles are blended with tapered weights to avoid seams. D4
predicts four rotations and their reflected versions, inversely transforms
the outputs, and averages probabilities. It increases inference cost; the
single-model candidate records actual end-to-end timing.

Connected components give a simple instance baseline. Hysteresis requires a
confident seed while retaining lower-confidence attached branches. Watershed
can separate nearby filaments but can also split one filament too aggressively.
The new fragment-grouping option uses a closed mask only to decide which
components share an identity, then discards all added pixels. It can still
merge distinct nearby objects, so held-out validation and mask inspection
are required before use.

The optional quality regressor estimates whether a proposed instance resembles
a valid match from image-only probability, contrast, and shape measurements.
It is trained using fold-0 OOF proposals with temporal-group cross-validation.
It cannot recover missed pixels and may discard true small filaments; its
held-out quality must justify using it. The deployed JSON trees require no
training labels or scikit-learn dependency.

## Reading the results

Public-notebook PQ pools matched IoU and TP/FP/FN across annotation views.
Our standard one-to-one PQ diagnostic uses an assignment instead. An image
union's Dice can improve while PQ falls if instance identities are wrong.
The grid-MIoU value remains a provisional diagnostic, not an official metric.

Fold 0 selects decoding parameters and is development evidence. A saved-map
subset from other folds is useful for screening, but cannot be relabeled as a
complete-fold estimate. The current final-ensemble estimate uses two held-out
members per fold without D4; the accepted test submission uses ten models and
D4. Their configurations differ, so local and public scores are not interchangeable.

The expanded CSV validator checks canonical COCO compressed counts, image IDs,
dimensions, nonempty masks, unique instance IDs, and zero overlap within each
image. The suffix identifies a prediction, not the corresponding ground-truth
filament. A valid file is necessary for scoring, but says nothing about mask quality.
