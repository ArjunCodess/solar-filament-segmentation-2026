# Solar filament segmentation, IEEE Big Data Cup 2026

Research implementation by Arjun Vijay Prakash. The best accepted candidate is the refined single ConvNeXt model with D4, Kaggle submission 57906671, public PQ 0.34. The earlier ten-member ensemble scores 0.33. A top-three result is not established.

The pipeline normalizes radial illumination, trains native-resolution segmentation models with full-disk context, and extracts nonoverlapping filament instances. Training uses only competition-approved segmentation masks. Independent annotation views stay in the same temporal fold.

Install requirements.txt in Python 3.12 with a compatible PyTorch build, then run `python -m pytest -q`. See docs/method_walkthrough.md and docs/competitive_campaign.md for methodology, scope, and experiment status.

To reproduce the 0.34 candidate, change into `candidates/refined`, install its pinned requirements in a Python 3.12 environment with compatible CUDA, set `SOLAR_TEST_IMAGES` to the official competition test_images directory, and execute `pipeline.ipynb`. Download Git LFS weights first using `git lfs pull`. That folder preserves the exact inference source, full-precision weights, frozen policy, environment, provenance, and submitted CSV. The root training source includes newer sampling options; use the candidate folder for exact historical inference.

Measured inference: 1852.5 seconds for 180 native-resolution observations on one Tesla T4, including D4 and map-cache writes. Local reference PQ 0.407945 covers complete fold 1 without D4, not the full five-fold D4 candidate.

The organizer-format report is `paper/organizer_report.pdf`; original IEEE working notes remain separately available. A paired sampling experiment is running. Final organizer-form delivery remains pending team-country details and final candidate selection. No rank or award is guaranteed.
