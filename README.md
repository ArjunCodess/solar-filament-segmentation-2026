# Solar filament segmentation, IEEE Big Data Cup 2026

Research implementation by Arjun Vijay Prakash. The accepted five-fold submission currently has public PQ 0.33. New experiments are in progress; this source release is not yet an organizer-ready final submission.

The pipeline normalizes radial illumination, trains native-resolution segmentation models with full-disk context, and extracts nonoverlapping filament instances. Training uses only competition-approved segmentation masks. Independent annotation views stay in the same temporal fold.

Install requirements.txt in Python 3.12 with a compatible PyTorch build, then run `python -m pytest -q`. See docs/method_walkthrough.md and docs/competitive_campaign.md for methodology, scope, and experiment status.

Weights and the final reproduction notebook will be included after model selection. The current source snapshot alone cannot reproduce the submitted model predictions. The four-page paper is working notes.
