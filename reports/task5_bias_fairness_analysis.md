# Bias & Fairness Analysis

## Scope and decision context

This analysis audits the locked Task 4 Random Forest on the 885-record held-out test set. A `Dropout` prediction at the saved 0.23 probability threshold is treated as the adverse selection event for fairness measurement. In practice, an alert can create both benefit (earlier support) and harm (stigma, unnecessary contact, or punitive misuse), so alert-rate parity alone does not determine fairness.

Gender, age, and international status were excluded from the model inputs and retained only for auditing. Financial-risk and scholarship indicators are audited as incomplete socioeconomic proxies. Race is absent from the dataset. Nationality or international status is not treated as race.

## Model explanations

Tree SHAP was computed for all held-out records. The leading grouped drivers of modeled dropout risk were first_sem_progress_index, first_sem_approval_rate, Tuition fees up to date, Curricular units 1st sem (approved), financial_risk_flag. Positive SHAP values raise modeled dropout probability relative to the model's expected value; negative values lower it. Grouped importance sums one-hot component contributions, so high-cardinality variables such as course and application mode require cautious comparison. These explanations describe model behavior, not causal effects or recommended interventions.

PDP and ICE curves were generated for first-semester progress and approval rate. They show the model's average and individual response when each feature changes while other inputs remain fixed. Because engineered academic features are related to their source variables, these plots are diagnostic and should not be interpreted as feasible or causal interventions.

## Fairness metrics

The audit reports observed dropout rate, alert rate, precision, true-positive rate (dropout recall), false-positive rate, calibration, demographic-parity difference, adverse-alert disparate-impact ratio, equal-opportunity difference, and equalized-odds gaps. Reference groups are chosen explicitly for stable comparison, not labeled inherently privileged.

- **Gender:** female alert rate 36.0%; male alert rate 55.6%. Female dropout recall is 79.1%; male recall is 88.3%. The maximum TPR/FPR gap is 9.1%.
- **Age:** under-25 alert rate 34.6%; age-25-or-older alert rate 66.2%. The maximum TPR/FPR gap is 17.5%.
- **Financial-risk proxy:** no-immediate-risk alert rate 32.6%; financial-risk-indicated alert rate 85.5%. The adverse-alert rate ratio is 2.63.
- **International status:** the international subgroup has only 24 test records. Its metrics are reported but are too unstable for a strong conclusion.

Demographic-parity gaps partly track real outcome-rate differences in this historical dataset. That does not prove the alert policy is fair: historical outcomes may themselves reflect unequal access, financial constraints, or institutional processes. Equalized-odds components and the practical consequences of false positives and false negatives deserve more weight for a supportive intervention.

## Imbalance, leakage, and overfitting

The minority `Enrolled` class represents about 18% of records and dropout about 32%. Task 4 used stratification, class weighting, macro metrics, dropout-specific metrics, and a fixed-capacity capture metric rather than accuracy alone.

Temporal leakage was reduced by excluding all second-semester academic variables. Preprocessing, feature selection, model selection, and threshold selection used only the training partition. Task 5 uses the held-out data only to audit the already locked model.

The held-out-minus-training gaps range from -0.023 to -0.004 across the reviewed metrics. This suggests a modest generalization gap in this split, not proof that overfitting is absent. External and temporal validation are still required.

## Data and ethical limitations

- The data comes from one Portuguese higher-education institution, limiting external validity.
- Race is unavailable. The audit cannot assess racial fairness, and nationality is not a valid replacement.
- Gender is encoded as a binary field and may not reflect identity or nonbinary students.
- International and educational-special-needs groups are small, producing unstable estimates.
- Financial-risk and scholarship indicators do not measure income, wealth, caregiving, disability-related costs, or access to support.
- Historical dropout labels can encode institutional and structural inequities.
- SHAP, PDP, subgroup associations, and fairness gaps are descriptive rather than causal.
- A threshold that improves recall increases false-positive outreach and reduces overall multiclass performance.

## Mitigation plan

| priority | action | risk_addressed | validation_before_use |
| --- | --- | --- | --- |
| 1 | Use alerts only for supportive outreach with human review, documented reasons, and an appeal path. | False positives, stigma, and automated adverse decisions | Confirm no alert is used for exclusion, discipline, grading, or financial penalties. |
| 2 | Collect better representation data under consent and governance, including self-described gender and race/ethnicity where lawful. | Binary gender field and missing race prevent a complete audit | Publish purpose, retention, access, minimum cell size, and deletion rules before collection. |
| 3 | Evaluate reweighting or group-aware resampling inside nested cross-validation. | Unequal false-negative and false-positive rates | Compare recall, precision, calibration, and equalized-odds gaps on a new validation cohort. |
| 4 | Calibrate dropout probabilities using training-only cross-validation. | Probability miscalibration across risk ranges and groups | Check Brier score and reliability curves overall and by sufficiently large group. |
| 5 | Consider group-specific thresholds only after legal and ethical review. | Recall gaps that a single threshold cannot resolve | Tune on validation data, document the fairness objective, and test for new precision or burden disparities. |
| 6 | Monitor performance, alert rates, TPR, FPR, calibration, and data drift by cohort and group. | Temporal drift and degradation after deployment | Set minimum sample sizes, confidence intervals, escalation thresholds, and rollback ownership. |

## Decision

The model should not be deployed for automated adverse decisions. It is suitable only for a controlled, non-punitive support pilot after review of the flagged group disparities, training-only mitigation experiments, probability calibration, human oversight, an appeal process, and prospective monitoring. A new cohort should be reserved for validating any mitigation or threshold change.
