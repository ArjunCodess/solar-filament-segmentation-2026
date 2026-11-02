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
