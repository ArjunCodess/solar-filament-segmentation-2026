# Organizer reference material

Source: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Retrieved from the public Kaggle kernel API on 2026-10-28, version 6. `self_evaluation_api.json` preserves API metadata and original notebook text, `self_evaluation.ipynb` is the original notebook, and `self_evaluation_source.py` concatenates code cells for inspection. These are organizer-authored reference materials, not original code in this project. Preserve source attribution and review the notebook's license before redistributing it.

Parity tests execute only the scoring function definitions, not notebook cells or downloads. The public notebook uses all IoU > 0.5 pairs and micro aggregation across annotator-image entries. Standard one-to-one PQ is also available as `standard_pq` for analysis. The hidden server includes additional edge-case checks that are not specified by this notebook, so server parity is not asserted.
