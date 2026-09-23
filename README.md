
# CNC Tool Wear Detection — Project Notes

## Task
Predict whether a CNC mill's cutting tool was **worn** or **unworn**, using
the motor sensor readings recorded while it cut 18 wax test parts.
(Dataset: University of Michigan SMART Lab, via Kaggle.)

## Approach
- **Unit of prediction:** every individual sensor reading (one row = one
  100ms timestep) during active cutting ("Layer 1 Up" through "Layer 3
  Down"), labeled with its parent experiment's tool condition.
- **Model:** Random Forest classifier (`scikit-learn`), 200 trees.
- **Evaluation:** `GroupKFold` cross-validation, grouped by experiment
  number (6 folds of 3 experiments each). This is the important part —
  see below.

## Why the split method matters
This dataset has only 18 experiments but ~1,000+ rows each. If you split
rows randomly into train/test, rows from the same experiment leak across
both sides of the split (they're 100ms apart and nearly identical), so
accuracy looks artificially high (often 95%+ in public notebooks that
make this mistake). Splitting by **experiment** instead — so a whole
experiment is either entirely train or entirely test — gives an honest
estimate of how the model performs on a genuinely new machining run.

## Result
- **83.9% cross-validated accuracy**, vs. a 54.1% baseline from just
  guessing the majority class (worn — 10 of 18 experiments used a worn
  tool). See `outputs/confusion_matrix.png`.
- The most influential features were the Z-axis (depth of cut) position
  and current-draw signals, plus X/Y output current — physically
  sensible, since a worn tool needs more force to cut, which shows up as
  higher motor current. See `outputs/feature_importance.png`.

## A note on the alternative approach (worth mentioning in your report)
The "obvious" alternative is to summarize each of the 18 experiments
into one row (e.g. mean/std of each sensor) and classify at the
experiment level. I tried this first — it performs at or below the
majority-class baseline (measured ~30-45% accuracy across a few
variations), because 18 samples is too few relative to the number of
sensor features, so the model overfits noise. This is a genuine,
reportable finding: it shows *why* the row-level, group-validated
approach above is the sounder design for this dataset, not just a
different way to get the same answer.

## Possible extensions (if you want to go further)
- Predict `passed_visual_inspection` or `machining_completed` instead
  (also in `train.csv`) using the same pipeline.
- Try other models (logistic regression, gradient boosting) and compare.
- Treat each experiment as a true time series (e.g. features from
  `tsfresh`, or an LSTM) instead of independent rows.
