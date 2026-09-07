# Student Dropout Risk Capstone

An end-to-end machine learning capstone for identifying university students who may benefit from early, non-punitive support after first-semester results become available.

**Public repository:** https://github.com/sudo1234master/student-dropout-risk-capstone

The project uses the public UCI **Predict Students' Dropout and Academic Success** dataset and predicts one of three outcomes: `Dropout`, `Enrolled`, or `Graduate`. The operational risk score is the model's predicted probability of `Dropout`.

## Headline result

The selected Random Forest uses a training-only tuned dropout-alert threshold of **0.23**. On the locked 885-record test set it achieves:

| Measure | Predefined target | Held-out result |
|---|---:|---:|
| Dropout recall | At least 75% | 83.8% |
| Dropout precision | At least 50% | 62.6% |
| Dropouts captured in highest-risk 30% | At least 60% | 72.9% |

The 30% figure is an explicit planning scenario, not a known institutional staffing limit. It is close to the source dataset's 32.1% dropout prevalence, supports comparison with random selection, and must be replaced with verified local outreach capacity before deployment.

## Responsible-use position

This model is **not recommended for automated adverse decisions**. A risk alert should only trigger respectful human review and offers of support. It should never determine admission, discipline, financial aid removal, or exclusion from services. The fairness audit identifies material subgroup disparities and limited external validity, so any operational use requires a controlled pilot, appeal and correction routes, probability calibration, group monitoring, and validation on a new local cohort.

## Repository structure

```text
.
├── configs/          # Saved Task 4 and Task 5 configuration
├── data/
│   ├── dictionary/   # Data overview and field dictionary
│   ├── external/     # UCI metadata
│   ├── processed/    # Fixed Task 3 development and test datasets
│   └── raw/          # Public UCI source CSV
├── models/           # Saved preprocessing, model, and SHAP artifacts
├── notebooks/        # Executable Task 3-6 notebooks
├── presentations/    # Stakeholder communication artifacts
├── reports/          # Task reports, tables, figures, and final report
└── src/              # Reusable analysis, modeling, audit, and validation code
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/run_pipeline.py --validate-only
```

To regenerate Tasks 3-5 from the public source data and then validate the outputs:

```bash
python src/run_pipeline.py
```

To regenerate the outputs and execute the three analytical notebooks:

```bash
python src/run_pipeline.py --execute-notebooks
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full local setup and execution instructions.

## Method summary

- Fixed stratified 80/20 development-test split with random seed 42
- Prediction point at the end of the first semester
- Six second-semester academic variables excluded to prevent temporal leakage
- Audit-only demographic fields excluded from model inputs
- Training-only preprocessing, feature selection, grid search, model selection, and threshold tuning
- Five-fold stratified cross-validation across Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting, with a dummy baseline
- Locked held-out evaluation, SHAP explanations, PDP/ICE checks, subgroup metrics, and intersectional fairness screening

## Dataset

Realinho, V., Machado, J., Baptista, L., and Martins, M. V. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See `DATA_LICENSE_AND_ATTRIBUTION.txt` and `CITATION.md` for attribution details.

## License

Original repository code is released under the MIT License. The UCI dataset remains under its separate CC BY 4.0 license. Reports, model artifacts, and third-party materials retain any rights and conditions stated in their source documentation.
