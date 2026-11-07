# Solar filament improvement and submission plan

Prepared November 7, 2026. This document authorizes no new execution by itself. This turn is research and planning only: no model changes, training, inference, cloning, prediction uploads, or organizer-form delivery. The dates below are proposed execution windows, not scheduled jobs.

## Decision

Keep the accepted 0.34 public-PQ candidate as the fallback. Prioritize recovering missed instances through object-conditioned segmentation and a small-object specialist, then test whether a complementary public recipe helps. Do not spend the remaining week enlarging the existing ensemble without evidence that its errors differ.

Winning is uncertain. The competition judges morphology, feasibility, and reproducibility as well as quantitative results; there is no verified score that guarantees an award. Our plan must improve actual masks and complete final delivery even if further training fails.

## What the competition actually requires

Challenge 02 has one advertised solar-filament task and first-, second-, and third-place awards. I found no separate solar efficiency, morphology, or paper award track. Kaggle predictions and the organizer's report/repository form are complementary requirements for the same final evaluation. The published rubric allocates 70% to quantitative comparison and 30% to qualitative comparison. A polished report cannot replace accurate predictions.

The challenge-specific final report/solution date is November 15, 2026. One indexed official page specifies 11:59 pm CDT, but the timezone wording needs confirmation before relying on an exact converted timestamp. Finish on November 14 instead. November 20 is the umbrella conference camera-ready date, not a safe extension for this challenge. The official overview requests a four-page report using its linked template, a public repository, pinned requirements, and an end-to-end notebook. The existing organizer-template report and separate IEEE notes should remain distinct. Re-check current form fields, template instructions, and weight-release requirements before delivery. Sources: [official overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates), [organizer remarks](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503).

The [IEEE Cup page](https://bigdataieee.org/BigData2026/cup/) lists six challenges. Other platforms include Codabench for UrbanTwin's LUMPI/V2X tracks and OpenCity; TrafficFlowBench and CarbonGlobe use Kaggle; FinReason uses task-specific platforms. These require separate solutions. FinReason's listed final-system and working-note deadlines have already passed. The umbrella page still describes MIoU for solar segmentation, whereas the challenge-specific page documents the August switch to PQ. Use the specific challenge's current instructions. Do not claim a separate win by relabeling the qualitative rubric as another track.

## Verified starting point and unresolved state

- Accepted candidate 57906671 has public PQ 0.34, with exact source and weights under `release/public/candidates/refined`. Its recorded 180-image D4 inference time is 1852.53 seconds on a T4. The local 0.407945 result covers fold 1 without D4, so it does not validate the exact submitted inference configuration.
- The approved competition manifest has 707 physical training images, 1,154 annotator views, 8,199 annotations, and 180 test images. The larger 1,593-image MAGFiLO research release is not the approved competition training split. Do not import its additional ground truth.
- Existing complete-view failure analysis found 2,240 false negatives and 3,729 false positives for the calibrated ensemble. Of the missed annotations, 785 had no predicted skeleton coverage and 379 intersected multiple predicted fragments. These measurements identify missed objects and incorrect object identity as priorities; they are not measurements of every later candidate.
- Quality filtering improved local PQ but failed the declared morphology gate. Mask-only continuation missed all 22 small fold-1 objects. Do not silently revive either as a final recipe.
- `sampling-v4` and `instances-v5` were last recorded running on November 2. Current status is unverified: the old credential delivery file is absent. Retrieve status and lightweight outputs before launching replacements. No outcome has been established for either experiment.
- Earlier folds have already influenced development. Calling them untouched test sets would be incorrect. Last recorded 42 passing tests are historical evidence, not a new test run in this planning turn.

Local evidence: `docs/resume_current.md`, `docs/competitive_campaign.md`, `docs/failure_audit_20261102.json`, and `docs/data_summary.json`.

## Public solutions to investigate

See [the source register](public_solution_register_20261107.md) for links, licenses, and limitations. First screen Farnoosh's newer `run3` decoder ideas on existing maps where they differ from our current extraction. The main new training target is HeShen-1's seed-conditioned detector/refiner, followed by a lightweight native instance segmenter. Felipe's implementation offers a second-stage reference, but its July 0.71 is not a current PQ benchmark. A later August ledger claims 0.54 for a different candidate whose full reproduction is unavailable; that deserves investigation, not automatic adoption. None has been run by us in this planning turn.

For each imported artifact, record the commit, license, original author, checkpoint hash, training data and split, preprocessing, runtime, and metric date. Retain upstream notices. If copying rights are absent, reproduce the published method independently rather than vendoring its code. Public availability of weights does not establish acceptable training provenance.

## Execution order

1. **Recover existing experiments and align evaluation.** Restore Kaggle access through secure credential delivery if required, inspect both finite jobs, and download only lightweight summaries first. Recompute their sufficient statistics and review their masks. Re-evaluate the preserved fallback with the exact proposed D4, tiling, normalization, and disjoint-instance decoder on complete held-out views. This closes the current local-versus-submission configuration gap before comparisons.

2. **Freeze the comparison protocol.** Pin the official public self-evaluation notebook version and the local implementation. Preserve both its all-qualifying-pairs convention and standard one-to-one PQ as explicitly named diagnostics. Evaluate after overlap resolution and RLE round-trip, with every annotator view kept separate. Group identical physical images and neighboring observations together. Use calibration data only for thresholds; freeze each policy before complete-fold evaluation. Evaluate a longer temporal-gap sensitivity check without pretending it creates a new untouched test set.

3. **Screen reusable decoding, then reproduce one object-centric recipe.** Farnoosh's `run3` branch adds hysteresis and a ridge-informed component selector. Our decoder already has confident seeds and area filters, so identify the actual difference before rerunning another threshold search. Use cached calibration maps for a bounded comparison, and retain the existing morphology gate because a prior learned filter already failed it. For new training, start with a seed-conditioned crop refiner: image, contrast-enhanced image, and target seed/box identify which filament to retain when neighboring structures enter the crop. Train with approved masks on our split and comparable compute. Compare against `instances-v5` before introducing another expensive detector. Use the external checkpoint only if its provenance and evaluation separation are demonstrable; a checkpoint trained on our validation images cannot yield OOF evidence. Otherwise retrain the recipe. Separate a faithful upstream reproduction from a rules-compliant variant whenever initialization must change.

4. **Build a recall specialist only if the failure audit supports it.** Use native-resolution positive crops balanced by instance rather than area, with training-only dark distractors. Keep full-disk context to reject sunspots and atmospheric artifacts. Compare sampling alone before adding another loss or architecture. For long objects, use context-expanded or multi-scale crops and measure crop truncation explicitly. Do not impose an arbitrary per-image instance cap or force one prediction on every blank image.

5. **Test annotation handling as one controlled change.** Compare the existing per-view sampling with a representative annotator target and agreement-weighted union supervision on the training portion only. The latter can retain faint structures marked by one expert while reducing the influence of disputed boundaries. Never union annotator masks during official validation. Do not treat human agreement as a hard accuracy ceiling; it measures ambiguity and depends on the scorer. Stop this branch if it fails to improve the fixed comparison.

6. **Repair identity without manufacturing foreground.** Compare the current decoder with seed-conditioned instances and conservative fragment association based on distance, tangent direction, intervening image evidence, and predicted object membership. Test a specific hybrid: use object proposals for identity and our native-resolution ConvNeXt maps for fine foreground boundaries. Measure proposal-gating recall losses so missed detections do not erase valid semantic predictions. Keep disconnected pixels under one instance when justified. Resolve overlapping proposals by calibrated pixel ownership, then measure the damage from ownership resolution. A plain union of all masks can create merges and is not our ensemble method.

7. **Run the requested ours/theirs/combined comparison.** Evaluate the preserved fallback, retrieved sampling model, retrieved object model, faithful public reproduction, adapted public recipe, and one small complementary fusion. Start with ours plus the strongest reproduced model. Match corresponding instances, choose or blend their masks using frozen calibration, and separately gate unmatched detections. Measure correct objects recovered versus false positives added. Test public-public fusion only if the first comparison shows complementary errors. Avoid a Cartesian product of every model, threshold, and TTA setting.

8. **Expand only the winner and freeze.** A complete-fold screening gain earns another temporal-fold check, not automatic submission. Expand the best supported recipe to remaining folds only if runtime permits, then compare the final smaller model against its ensemble. A clean fallback submission is preferable to missing final delivery while another run trains. Freeze model and decoder by November 13, complete reproduction and form delivery on November 14.

## What counts as an improvement

For every candidate, save per-view IoU sums, TP/FP/FN, pooled reference PQ, standard PQ, matched IoU, precision/recall, split/merge counts, skeleton coverage, small-object hit rate, and strata by contrast, object area, disk radius, date, and observing station when metadata permits. Report physical-image and annotation-view sample counts. Compare the same images and metric aggregation; do not compare one repo's mean-fold PQ with our pooled PQ as though they were equivalent.

Use paired temporal-block bootstrap intervals for final comparisons. They quantify held-out variation conditional on the trained models, not full training or selection uncertainty. Extend the existing promotion gate: at least +0.005 held-out PQ, skeleton coverage loss no greater than 0.01, and small-object hit-rate loss no greater than 0.02, plus visual review for systematic merges and omitted barbs. Prefer a positive paired interval before final promotion. Inconclusive evidence favors the simpler preserved candidate. The gate is a decision rule, not a promise of leaderboard gain.

Measure the exact inference recipe's wall time, peak memory, model count, and storage. Cache validation maps when affordable; keep large caches remote and inspect free space before downloads. Budget initial experiments against remaining GPU quota, then select a finite run list. No paid compute without an authorized budget and no hidden controller or automatic submission loop.

Preprocessing gets an isolated test only if the winning model's errors warrant it: raw/robust normalization versus radial correction plus CLAHE. Check limb contrast and noise amplification rather than assuming normalization always helps. Reserve EdgeAttNet-style edge guidance or a query-based instance model for a second round only if the prioritized experiments finish early. Do not retrain a large foundation-model stack solely because it is newer.

## Deliverables after execution

- A paired comparison ledger with candidate identity, source/weight hashes, split manifest, budget, exact inference settings, metrics, uncertainty, runtime, and promotion decision.
- A fixed qualitative panel selected by size/contrast/radius strata, including failures, showing raw input, independent expert views, fallback, reproduced public model, and final candidate at native-resolution barb crops.
- A final disjoint-mask CSV with independently decoded COCO RLE, valid image IDs, unique filament IDs, no empty masks, and no fabricated objects to force image coverage. Account for all 180 inputs, including genuinely empty outputs.
- A public release with the selected weights stored as required, pinned dependencies, a notebook that loads those weights, deterministic inference configuration, provenance, attribution, and a clean-machine reproduction record. Verify Git LFS downloads anonymously if LFS is used.
- A report following the organizer's current template and page instructions, with measured ablations, negative results, feasibility, limitations, and AI assistance disclosure. Keep IEEE working notes separate. Render and inspect the final PDF.
- A completed organizer form with the selected submission/reference, report and repository links, correct sole-member details for Arjun Vijay Prakash, India, and a saved receipt. Record accepted Kaggle submission status and final selection separately from form acceptance.

## Proposed week and user handoff

November 7-8: recover jobs, confirm requirements, freeze evaluation, and establish the exact fallback benchmark. November 8-10: one public reproduction and targeted sampling/seed-conditioning comparisons. November 10-12: confirm the strongest recipe on additional folds and test one complementary fusion. November 13: freeze candidate and finish inference. November 14: clean reproduction, report, Kaggle selection, organizer form, and receipt. November 15 is contingency time, not planned research time.

You do not need to choose architectures or operate experiments. To begin the implementation phase after this planning-only turn, say "Execute this plan." If Kaggle access still needs a private credential, I will request it through the secure secret tool. Before final delivery, review the short method explanation and confirm you can explain the pipeline; organizer guidance makes understanding the AI-assisted work relevant to eligibility. I will prepare the technical artifacts and submission details.
