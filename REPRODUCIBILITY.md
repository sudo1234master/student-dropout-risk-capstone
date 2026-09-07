# Reproducibility guide

## Tested environment

The committed notebooks were run locally with Python 3.13.9 and the exact package versions in `requirements.txt`. The modeling workflow uses random seed 42 and a fixed stratified 80/20 development-test split.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name student-dropout-capstone --display-name "Student Dropout Capstone"
```

Windows activation command:

```powershell
.venv\Scripts\activate
```

## Run the notebooks

Open Jupyter from the repository root, select the environment created above, and run the notebooks in numerical order:

1. `notebooks/task3_eda_feature_engineering.ipynb`
2. `notebooks/task4_model_implementation.ipynb`
3. `notebooks/task5_bias_fairness_analysis.ipynb`

Each notebook discovers the repository root automatically and can also be opened directly from the `notebooks` folder.

## Run from the command line

Regenerate Tasks 3-5 and validate all outputs:

```bash
python src/run_pipeline.py
```

Also execute and save the notebooks:

```bash
python src/run_pipeline.py --execute-notebooks
```

Validate the committed artifacts without retraining:

```bash
python src/run_pipeline.py --validate-only
```

Task 4 performs grid search and cross-validation and therefore takes substantially longer than Tasks 3 and 5.

## Leakage and evaluation controls

- Six second-semester variables are excluded from the end-of-first-semester prediction scope.
- The held-out test set is created before learned preprocessing and model development.
- Preprocessing, feature selection, hyperparameter selection, model selection, and alert-threshold selection use training data only.
- Task 5 audits the locked Task 4 model on the held-out test set; mitigation proposals are not tuned on that test set.

## Expected headline result

The committed Task 4 configuration selects a Random Forest and a dropout alert threshold of 0.23. On the 885-record held-out test set, the alert achieves approximately 83.8% dropout recall and 62.6% dropout precision. The highest-risk 30% of records capture approximately 72.9% of held-out dropouts.

