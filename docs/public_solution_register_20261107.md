# Public solution register

Researched November 7, 2026. Scores below are authors' reported results, not our reproductions or an independently refreshed leaderboard. No external code or weights were downloaded or executed in this planning turn. A license for repository code does not automatically cover checkpoints, dependencies, competition data, or copied notebooks.

## First reproduction candidate

[HeShen-1/filament-segmentation-2026](https://github.com/HeShen-1/filament-segmentation-2026) reports public PQ 0.35 and five-fold mean PQ 0.3593 for its single pipeline. It uses YOLO11m-seg at 1536, representative annotator targets, a seed-conditioned ResNet18 U-Net, hard-negative crops, and flip TTA. Its own experiments report correlated false positives from larger fusions and no transferable benefit from longer training. This supports testing target conditioning and detection diversity before increasing model count. These cross-validation scores are not directly comparable to ours without identical splits and aggregation.

Its [MIT license](https://raw.githubusercontent.com/HeShen-1/filament-segmentation-2026/main/LICENSE) covers original code and documentation. Weights are not bundled; Kaggle kernel dependencies describe the artifact chain. The optional i079 fusion member needs a separate model-card, data-provenance, and license check. The README itself identifies external-member contamination of local validation as a concern. Reproduce the training recipe on our split for reliable OOF comparison. Source: [README](https://raw.githubusercontent.com/HeShen-1/filament-segmentation-2026/main/README.md).

## Object pipeline and negative-result reference

[FELIPEACASTRO/solar-filament-2026](https://github.com/FELIPEACASTRO/solar-filament-2026) provides an object-centric detector/refiner pipeline and extensive experiments. Its advertised 0.71 comes from July's earlier scoring period and is not evidence that it beats our present PQ 0.34. Use architectural and reproduction details, not its old leaderboard number or historical scorer-specific tactics, to choose experiments.

Original work is [MIT licensed](https://raw.githubusercontent.com/FELIPEACASTRO/solar-filament-2026/master/LICENSE), but the license explicitly excludes third-party material and notes an unresolved redistribution item. Read [NOTICE](https://raw.githubusercontent.com/FELIPEACASTRO/solar-filament-2026/master/NOTICE) and the file-level inventory before reusing code. The README explicitly says weights are unpublished and the submitted CSV cannot be regenerated. The existing `instances-v5` experiment already tests a related independently implemented idea, so retrieve it before repeating that work.

Important later evidence: an [August 24 snapshot](https://raw.githubusercontent.com/FELIPEACASTRO/solar-filament-2026/master/analysis/omission_recovery_20260824/OFFICIAL_STATE_SNAPSHOT.json) records displayed public PQ 0.54, submission 55742270. Its [separate final plan](https://raw.githubusercontent.com/FELIPEACASTRO/solar-filament-2026/master/analysis/omission_recovery_20260824/FINAL_SUBMISSION_PLAN.md) identifies a different incumbent, inaccessible baseline weights, and no reliable OOF validation. This is a self-published historical receipt, not a currently verified rank or a reproduced clean-data result. Investigate its generating method and lineage before making any claim that it is transferable. Do not copy its submitted test masks.

## Lightweight native-instance control

[Wzy633/solar-filament-segmentation-2026](https://github.com/Wzy633/solar-filament-segmentation-2026) reports public PQ 0.31 and submission-legal fold-0 PQ 0.352585. Its YOLO26n-seg pipeline is a useful compute-efficient control, not an established improvement. The experiment log shows that permitting overlapping masks inflated local PQ and yielded an invalid submission, so evaluate only after disjoint ownership resolution. Avoid transferring its force-one fallback or ten-instance cap without evidence.

A reusable license and downloadable trained weights were not verified in the inspected repository. Independently implement the recipe if rights remain unclear, and inspect initialization provenance. Source: [experiment log](https://raw.githubusercontent.com/Wzy633/solar-filament-segmentation-2026/main/EXPERIMENTS.md).

## Cheap decoder screen and semantic baseline

[FarnooshMemari/solar-filament-segmentation](https://github.com/FarnooshMemari/solar-filament-segmentation) reports public PQ 0.32 and local PQ 0.389. It uses a compact scratch-trained U-Net, soft annotator targets, half-resolution images, and tiled inference. This is a possible runtime control; it is less compelling than our existing native-resolution model for thin structures. The README declares MIT code, but verify the license file at the pinned commit before copying. Trained-weight availability is unverified. Source: [README](https://raw.githubusercontent.com/FarnooshMemari/solar-filament-segmentation/main/README.md).

The separate [`run3` branch](https://raw.githubusercontent.com/FarnooshMemari/solar-filament-segmentation/run3/README.md) is more relevant than `main`: it reports an earlier public 0.34, then an unsubmitted multi-channel model with four month folds, EMA, radial/local-contrast/ridge channels, hysteresis, and a component picker. All-image OOF PQ is reported as 0.412 with rules and 0.415 with the picker. Threshold tuning uses the OOF corpus, so these figures still have selection bias. Test distinct decoder ideas on our saved maps first; retrain extra channels only if they address observed errors. The branch credits [tushhaaarrrr's public notebook](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-37), whose title alone does not establish score or provenance.

## Useful ideas without demonstrated superiority

[QiuweiLiu/solar-filament-segmentation-2026](https://github.com/QiuweiLiu/solar-filament-segmentation-2026) reports local tune-set PQ up to 0.3177 for a tiled U-Net with seed/boundary/spine heads and skeleton-endpoint association. It explicitly excludes checkpoints and caches. No copying license was verified. Its directional graph merge is a focused idea to compare with our distance-only grouping; its tune-set number is not a public benchmark.

[yadav-aryan18/solar-filament-segmentation-2026](https://github.com/yadav-aryan18/solar-filament-segmentation-2026) advertises SegFormer and Mask2Former. A credible matched-protocol score, reusable license, and checkpoint provenance have not been established in this audit. Keep it below the reproducible object-centric experiments in priority.

[The too.foo participant write-up](https://kaggle.too.foo/competitions/filament-segmentation-2026/) reports public PQ 0.32 and describes annotator-aware training, tile overlap, and small-filament oversampling. Its own local gains did not consistently transfer to Kaggle. Use this as evidence to isolate annotation weighting and sampling, not as proof that its ensemble is better. Accessible source, license, and reusable weights still need verification.

## Research worth a bounded follow-up

[EdgeAttNet](https://arxiv.org/abs/2509.02964) introduces image-derived edge guidance for barb-aware segmentation. Its research-dataset results are not a competition PQ result. It is a possible controlled decoder modification after detection and sampling experiments, not a reason to replace the entire pipeline.

[Graph learning for filament fragment association](https://arxiv.org/abs/2610.17291) treats fragments as nodes and their associations as learned edges. It motivates using geometric and image evidence for identity. Its setting and datasets differ; importing its annotations or assuming its graph generalizes to our single-image competition is unjustified.

## Reproduction record required for every selected source

Record upstream commit and URL, original authors, exact license and notices, dependency licenses, checkpoint checksum, training-data lineage, fold separation, preprocessing, training budget, and exact evaluation settings. Retrain on our approved split when external weights cannot establish separation. Compare standalone and adapted versions before any fusion, and retain the smallest model that wins the paired complete-fold comparison.

## Delegated audit provenance

Both requested research audits completed without implementation. Their findings were reviewed against primary sources; historical leaderboard ranks and the crossed-out Dice criterion were not carried forward as current facts.

- Grok 4.7, Cursor, high reasoning: <t3-thread://v2/f4e304be-5737-404a-95cc-347c66941a0c/thread:delegated-task:command%3Amcp%3A15bf1882-76be-493e-86ef-edec726ea971%3Adelegate-task%3Ad1122297-9acd-4eba-9d90-baa6bea9b764>
- Gemini 4 Argon, Antigravity, high thinking: <t3-thread://v2/f4e304be-5737-404a-95cc-347c66941a0c/thread:delegated-task:command%3Amcp%3A15bf1882-76be-493e-86ef-edec726ea971%3Adelegate-task%3A7eb18a6f-058b-44df-9c35-ddfaa4ae683d>
