# Competitive campaign, 2 November 2026

The current accepted submission is 57846445, public PQ 0.33. The live top three
are IONode Online at 0.70, Tempest at 0.64, and Sayandeep Dutta at 0.62. These
scores were retrieved through the official Kaggle CLI. They are leaderboard
references, not confirmed final award thresholds or verified descriptions of
those teams' methods. No private leading model has been accessed.

## What changes the strategy

The organizer's 724503 discussion says the board will be reviewed after the
deadline and that final evaluation includes visible morphology and feasibility.
The overview gives quantitative comparison 70% and qualitative comparison 30%,
with fragmentation, merging, documentation, code, and morphology considered.
Discussion 753816 explicitly prohibits overlapping predictions and confirms
undisclosed differences between the hidden evaluator and public notebook.
The local evaluator remains a reference implementation, not a verified replica
of the hidden server. We will not optimize against presumed guardrail behavior.

Public discussions do not disclose a reproducible top-three architecture. The
backbone thread 744765 contains participant claims about focal/Dice adjustment,
but its local validation protocols differ from ours. We treat those as ideas
to test, not comparable benchmark measurements.

## Experiments launched

1. `solar-filament-inference-v2` evaluates the fold-0-frozen calibrated ensemble
   on complete folds 1-4, then performs D4 inference on all 180 test images.
   Its policy comes from already completed calibration. Validate its CSV and
   submit it as a distinct candidate; preserve submission 57846445.
2. `solar-filament-shape-search-v3` searches 96 policies on saved fold-0 maps:
   two backbone weights, four foreground thresholds, two minimum areas, three
   fragment-grouping radii, and two mean-confidence thresholds. It freezes the
   winner before evaluating saved maps from folds 1-4. Grouping changes instance
   identity without adding foreground pixels. Per-instance diagnostics record
   size, contrast, radial position, skeleton coverage, fragmentation, and merging.
3. `solar-filament-multiscale-v3` starts both folds 0 and 1 from their existing
   refined ConvNeXt checkpoints. Each fold has two matched-budget variants,
   with the same initialization, seed, learning rate, physical-image balancing,
   and 640-pixel output crops drawn from 480-800 native pixels. One retains
   topology and auxiliary losses; the other uses mask Dice/focal alone. Each
   adds eight epochs of 800 samples. Both extraction policies are fitted on
   fold 0 before complete fold-1 comparison. This isolates the loss/supervision
   choice between the two new variants; comparison with the older model changes
   several things together, including inference tile size.
4. `solar-filament-quality-v3` fits shallow randomized regression trees on
   fold-0 OOF instance proposals. Targets are match-quality estimates derived
   only from approved masks. Selection uses grouped, out-of-sample predictions
   within fold 0; all annotators of a physical observation remain together.
   Folds 1-4 are evaluated afterward. Deployment uses a JSON tree representation,
   checked numerically against scikit-learn, so inference adds no sklearn dependency.
5. `solar-filament-efficient-v3` is packaged but initially could not start because
   Kaggle permits only two simultaneous GPU sessions. It uses the already
   validated refined fold-1 ConvNeXt, with its fold-0-fitted policy, to measure a
   single-model D4 candidate and build a source/weights/notebook archive. The
   original/refined complete fold-1 reference PQ values are 0.393660/0.407945.

The public notebook subset and complete-fold results have different scopes.
Saved maps cover only 16 physical observations per original experiment. They
screen decoder ideas but cannot establish complete-fold improvement.

## Verified results after retrieval

- Calibrated ensemble: full folds 1-4 reference PQ 0.383435 versus 0.380625
  for the submitted baseline. All 905 unique view rows were checked and PQ was
  independently recomputed. Its 1,682-instance CSV has zero format errors and
  zero overlapping pixels. Submission 57898046 completed at public PQ 0.33,
  matching the previously accepted public score rather than improving it.
- Shape search: selected mean probability >= 0.7 and grouping radius four,
  with equal backbone weights, foreground threshold 0.5, confident seed 0.65,
  and minimum area 256. Saved-map held-out PQ is 0.405923 versus 0.401225 for
  calibrated components and 0.398905 for submitted extraction on those views.
  Medium/large hit rates decline slightly, so the score alone does not justify
  promotion. No separate shape submission has been made.
- Quality filter: grouped fold-0 selection used 229 proposals in 13 temporal
  groups and chose minimum predicted quality 0.2. On the same 100 held-out
  views, PQ is 0.415530 versus 0.401225 for calibrated area filtering and
  0.387242 for the corresponding unfiltered, area-64 proposals. Each of folds
  1-4 improves relative to the calibrated baseline. All values were recomputed.
- Paired training: on all 215 fold-1 views, mask-only multiscale continuation
  reaches PQ 0.408136 and topology control 0.402440. The earlier refined model
  reaches 0.407945 with 512-pixel inference, so the tiny apparent gain also
  changes inference configuration. Mask-only misses all 22 small instances;
  topology control matches three. Mask-only skeleton coverage is also lower.
  Do not expand or promote this recipe from PQ alone.
- The isolated source release passes the expanded test suite. A dedicated
  private Git repository now exists at
  https://github.com/ArjunCodess/solar-filament-segmentation-2026 . It remains a
  source snapshot until selected weights, notebook, and final report are added.

Two explicit follow-up GPU jobs have been launched. `efficient-v3` measures the
single refined model and packages weights with code and notebook. `quality-full-v3`
checks all 905 held-out views and retains a quality-filter candidate only if PQ
improves by at least 0.005 while skeleton recall drops by at most 0.01 and small
instance hit rate drops by at most 0.02. Only then does that finite kernel run
D4 test inference. It does not submit automatically.

The next implemented sampling options are `--instance-balanced` and
`--dark-negative-probability`. The former samples target objects uniformly before
choosing a positive pixel. The latter uses centers prepared by
`scripts/prepare_dark_negatives.py`, excluding the dilated union of every
training annotator's foreground and excluding held-out observations entirely.
These options are tested but have no measured training improvement yet.

## Promotion rules

- Recompute pooled PQ from all per-view sufficient statistics. Compare policies
  on exactly the same annotation views and retain the one-to-one PQ diagnostic.
- A decoder or quality filter must improve held-out PQ and pass inspection for
  lost thin branches and inappropriate merges. A subset gain triggers full-fold
  checking, not immediate promotion as a final model.
- Expand paired training to folds 2-4 only after complete fold-1 evidence supports
  the selected recipe. Those folds become development evidence if used to select
  the recipe; they must not then be called untouched tests.
- Test submission ranking is supporting evidence. Do not repeatedly select tiny
  threshold changes against the public leaderboard.
- Prefer a feasible smaller pipeline if its predictions remain competitive.
  Report measured inference time, model count, and hardware rather than claiming
  an unmeasured speedup.

## Reproducibility and award delivery

The official final submission requires a four-page report, public Git repository,
requirements with versions, notebook, and model weights stored with the release.
The repository must become public by competition close. The report and final
Google form are separate from uploading a Kaggle prediction CSV. Do not claim
award eligibility is complete until each required item has actually been delivered.

The original v1 source remains under `release/v1`. Uploaded experiment scripts
are retrieved under `release/experiments` so later local edits cannot silently
change an experiment's recorded identity. No schedules, agents, or controller
loops are enabled. Remote kernels are explicitly launched finite experiments.

## Sources read

- Current challenge and rubric: https://www.kaggle.com/competitions/filament-segmentation-2026/overview
- Organizer remarks: https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503
- Multiple annotators and overlap clarification: https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753816
- Participant backbone discussion: https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/744765
- 2018 Data Science Bowl first-place write-up: https://www.kaggle.com/competitions/data-science-bowl-2018/writeups/ods-ai-topcoders-ods-ai-topcoders-1st-place-soluti
  It documents pretrained encoder/decoders, mask/separation outputs, random
  scale augmentation, D4, probability averaging, and learned proposal-quality
  regression. The new experiments borrow those ideas, not external labels.
- Hu et al., A Modern ConvNet for Solar Filament Detection, arXiv:2607.24525v1:
  https://arxiv.org/html/2607.24525v1 . Its method combines multiscale semantic
  segmentation, DenseCRF refinement, and DBSCAN grouping. It explicitly discusses
  ambiguity in fragment integration from a solitary H-alpha image. Its datasets
  and semantic metrics differ from this competition; none of its labels were
  imported. Our morphological grouping is not an implementation of DBSCAN or
  DenseCRF, and our model is not a reproduction of MORDEN.

## Completed follow-up, November 2

The portable refined ConvNeXt candidate, submission 57906671, completes with public PQ **0.34**, improving the original and calibrated ten-member ensembles at 0.33. Its exact CSV is `submissions/efficient_refined_v3.csv`, SHA256 `6b69e5d938aff2bf8fa087fa17eef645c0f02212c226b186993e36787d17c9a2`. There are 1465 masks over 177 observations; all 180 identities/dimensions and disjoint canonical RLE masks pass independent validation. Measured D4 inference takes 1852.53 seconds on one Tesla T4. The package preserves exact full-precision weights, source, notebook, policy, environment, and provenance. Weight hash and strict model loading plus a finite three-head CPU forward pass were independently verified. Root training sources have since added sampling options; historical candidate source remains unchanged.

Complete quality-filter validation covers the same 905 views and independently reproduces PQ 0.3834346037 for calibrated components, 0.3899634796 for shape grouping, and 0.3934529131 for the learned filter. A 10000-draw paired temporal-block bootstrap over 264 blocks gives quality-filter PQ gain 95% interval [0.007984, 0.012034]. It conditions on trained models/frozen policies and excludes retraining/selection uncertainty. The filter fails the predeclared skeleton-coverage gate and generates no test CSV. Medium-instance skeleton recall drops from 0.72683 to 0.69791; this is a morphology regression despite higher PQ.

A new finite two-GPU `sampling-v4` kernel compares matched eight-epoch continuations from refined checkpoints in folds 0 and 1. Both use physical-image balancing, fixed-scale 640-pixel crops, unchanged topology and auxiliary losses, and seed 342. The targeted recipe chooses foreground instances uniformly and uses 20% dark-negative crops, mined outside a dilated union of all training annotators. No held-out image or target is used for mining. Policies fit fold 0 before complete fold 1 scoring. It does not automatically infer test data or submit. No outcome is yet available.

A separate three-page report now follows the organizer's supplied ACM-format template, preserves fixed opening text and acknowledgment, and includes a whole-pipeline figure, qualitative examples, uncertainty, public scores, morphology regressions, and AI-assistance disclosure. The requested IEEE notes remain intact. All 38 implementation tests pass. Final organizer-form delivery still requires accurate team/country information.

## Object detection and instance refinement, November 2

Country and membership are confirmed by the user: India, sole member Arjun Vijay Prakash. The organizer form remains unsubmitted while model improvements are evaluated.

A fresh audit of all 905 calibrated-ensemble views finds 4085 qualifying pairs, 3729 false positives, and 2240 false negatives. Among missed annotations, 785 have zero predicted skeleton coverage and 379 overlap multiple substantial predicted fragments. Only 105 predicted objects overlap multiple annotations. Matched-pair mean IoU is 0.66357. An oracle that removes every unmatched prediction, without changing masks, would score local reference PQ 0.52079. An oracle that perfects all matched boundaries while retaining the same detection counts would score 0.57783. These are separate ground-truth-dependent diagnostic scenarios, not deployable algorithms or hidden-server estimates. Filtering alone cannot fix missed material or fragmentation. Details are in docs/failure_audit_20261102.json.

A public participant report describes a FasterR-CNN detector followed by a per-box segmenter: https://raw.githubusercontent.com/FELIPEACASTRO/solar-filament-2026/master/report/TECHNICAL_REPORT.md . Its reported 0.71 is dated July 27 and the same report records a scorer revision announced August 13; it is not evidence of a current comparable leaderboard score. We borrow the object-centric pipeline idea without importing its checkpoints or annotations. Related primary sources include the torchvision FasterR-CNN/MaskR-CNN documentation, MaskR-CNN arXiv:1703.06870, and the nnU-Net Revisited training-recipe study arXiv:2404.09556. The latter motivates testing a conventional Dice/BCE object refiner rather than adding more topology terms.

Implemented src/instance_model.py contains a FasterR-CNN ResNet50FPN detector with elongated thin-object anchors and a separate ResNet18 U-Net refiner. Only ImageNet classification initialization is used; detector heads are random and COCO detection weights are excluded. Training boxes are derived solely from approved segmentation masks. The detector sees a full-disk 1024-pixel representation; the refiner sees a jittered object crop resized to 512 pixels, with one specific instance as target and neighboring objects as background. It can keep disconnected pieces in one instance. Overlapping output proposals compete by per-pixel confidence to guarantee disjoint masks.

The finite `instances-v5` job trains independent temporal folds 0/1 on two T4s. Each detector gets 24 epochs of 480 samples; each refiner gets 20 epochs with up to 800 positive crop attempts per epoch. The last epoch is fixed in advance. Physical-image sampling is balanced, annotator views are selected within each observation, and held-out physical hashes are excluded. Fold 0 calibration selects proposal score, mask threshold, and minimum area on 16 physical images before scoring all 215 fold-1 views. Fold 1 has already been used for development; this is a recipe comparison, not a new untouched test set. There is no test inference or automatic submission.

Four additional tests cover disconnected instance identity, overlap resolution, box/image augmentation alignment, crop targets and finite gradients, and actual detector/refiner CPU forwards plus detector backward. All 42 tests pass. The existing 0.34 candidate and its exact source remain unchanged.


## Implementation and recovered experiments, November 7

Both v4 sampling and v5 object jobs completed. Independent recomputation matches all 215 identical fold-1 annotation views: sampling control 0.4180745031; targeted sampling 0.3926889476; instances-v5 0.3253240537. Refined reference 0.4079446075 uses different training/tiling. The matched targeted-minus-control delta is -0.0253855555, with 10000 paired temporal-block draws over 65 groups giving 95% interval [-0.0360241354, -0.0144383717]. Targeted skeleton recall improves .71705 to .76199 and small hits 3/22 to 5/22. It fails the PQ gate and is not promoted. scripts/audit_sampling_v4.py reproduces the weighted morphology audit; scripts/compare_candidates.py requires identical views and marks missing gates not evaluable.

Pinned Farnoosh run3 commit 728094a1780673db748428857ccf016b822ca371 was fetched with its MIT notice, and 20 upstream decoder/RLE tests passed. Public decoder v6 version2 selects among 16 settings on 23 fold0 saved views, then scores 24 fold1 saved views: baseline PQ .4297252636, upstream decoder .4647203052. This is a screen, not complete-fold evidence or an official score. Version1 used only four qualitative views and is superseded as screening evidence. No per-image instance cap is used. Exact sources and hashes are archived separately under release/experiments/public-decoder-v6-v1 and -v2.

Opt-in target box/seed conditioning adds a third refiner input channel, with two-channel compatibility preserved. Neighboring-target and forward/backward tests pass. Finite instances-v6 version1 is running with fold0 policy calibration and complete fold1 evaluation; no automatic test inference or submission. Finite exact-validation-v6 version1 separately evaluates refined without D4, refined with D4, and the sampling-control model with D4, including per-instance morphology outputs. These jobs have no measured results yet. Retrieve status before launching replacements. All 50 root tests pass after delegated integration. The accepted 0.34 candidate remains unchanged and final organizer-form delivery remains pending.
