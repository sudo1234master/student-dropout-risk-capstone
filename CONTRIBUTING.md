# Contributing

This repository is a capstone submission, but reproducibility fixes and documentation improvements are welcome.

1. Create a branch for the proposed change.
2. Keep the fixed train/test split and random seed unless the study design is intentionally revised.
3. Do not tune models, thresholds, or mitigations on the held-out test set.
4. Run `python src/run_pipeline.py --validate-only` before submitting a change.
5. Document any change that could affect performance, fairness, or the intended non-punitive use of the model.

Please do not add personally identifiable student data or use the model for automated adverse decisions.

