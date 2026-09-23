r"""
=========================================================================
 CNC TOOL WEAR DETECTION - AI in Manufacturing course project
=========================================================================

WHAT THIS SCRIPT DOES, IN PLAIN ENGLISH:

Every time this CNC mill cuts a part, it records ~48 sensor readings
(position, velocity, current, voltage... for the X, Y, Z axes and the
spindle) every 100 milliseconds. We have 18 of these recordings
("experiments"), and for each one we know whether the cutting tool used
was WORN or UNWORN (that's in train.csv).

Our job: train a model that looks at the sensor readings and predicts
whether the tool was worn or not. This is called SUPERVISED
CLASSIFICATION - "supervised" because we already know the right answers
for our 18 experiments and use them to teach the model, "classification"
because the answer is a category (worn / unworn), not a number.

THE ONE IDEA YOU MUST UNDERSTAND BEFORE ANYTHING ELSE:
--------------------------------------------------------
Each experiment file has ~1000-2200 rows (one per 100ms). It's tempting
to treat every row as a separate training example - that gives you
~17,000+ rows instead of just 18! But rows from the SAME experiment are
almost identical to each other (they're 100ms apart). If you randomly
shuffle all rows and then split into train/test, some rows from
experiment #7 end up in training and OTHER rows from experiment #7 end
up in "testing" - so the model isn't really being tested on anything
new, it's basically seen the answer already. Your accuracy will look
amazing (95%+) and be completely misleading.

The fix: whenever we split data into train/test, we make sure ALL rows
from a given experiment go together, on the same side of the split.
That's what "GroupKFold" does below - it's the single most important
technique in this script. Skipping it is the #1 mistake people make with
this exact dataset (search "CNC tool wear kaggle" and you'll find many
notebooks with 95%+ accuracy that made exactly this mistake).

HOW TO RUN THIS:
  1. Make sure you've followed the setup steps (venv + pip install, see
     the guide you were given alongside this file).
  2. Save this file as:  D:\AIiM project\main.py
     (i.e. in the SAME folder that contains your "datasets" subfolder)
  3. In VS Code, open a terminal and run:  python main.py
=========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of trying to pop up a window
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================================================================
# STEP 0: SETTINGS
# =========================================================================
# This assumes main.py sits next to your "datasets" folder, e.g.:
#   D:\AIiM project\main.py
#   D:\AIiM project\datasets\train.csv
#   D:\AIiM project\datasets\experiment_01.csv ... experiment_18.csv
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "datasets"
OUTPUT_DIR = THIS_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Can't find a 'datasets' folder next to this script at: {DATA_DIR}\n"
        f"Either move main.py so it sits next to your datasets folder, "
        f"or edit DATA_DIR above to point at the right place."
    )


# =========================================================================
# STEP 1: LOAD THE LABELS (which experiments used a worn vs unworn tool)
# =========================================================================
print("STEP 1: Loading labels from train.csv ...")
labels = pd.read_csv(DATA_DIR / "train.csv")
labels.columns = labels.columns.str.strip()  # tidy up column names just in case
labels["y"] = (labels["tool_condition"] == "worn").astype(int)  # worn=1, unworn=0

print(labels[["No", "tool_condition"]].to_string(index=False))
print(f"-> {labels['y'].sum()} worn, {(1 - labels['y']).sum()} unworn "
      f"experiments out of {len(labels)}\n")


# =========================================================================
# STEP 2: LOAD ALL 18 SENSOR TIME-SERIES AND STACK THEM TOGETHER
# =========================================================================
print("STEP 2: Loading the 18 experiment sensor files ...")
all_experiments = []
for exp_no in range(1, 19):
    filepath = DATA_DIR / f"experiment_{exp_no:02d}.csv"
    df = pd.read_csv(filepath)
    df["No"] = exp_no  # tag every row with which experiment it came from
    all_experiments.append(df)

data = pd.concat(all_experiments, ignore_index=True)
data = data.merge(labels[["No", "y"]], on="No")
print(f"-> Combined into one table: {data.shape[0]} rows x {data.shape[1]} columns\n")


# =========================================================================
# STEP 3: KEEP ONLY THE ROWS WHERE THE TOOL IS ACTUALLY CUTTING
# =========================================================================
# The Machining_Process column tells us what the machine was doing at each
# instant. Rows like "Starting", "Prep", "end" or "Repositioning" are the
# spindle moving through the air, not cutting - they won't tell us much
# about tool wear, so we drop them and keep only "Layer 1 Up", "Layer 1
# Down", "Layer 2 Up", etc. (the six actual cutting stages).
print("STEP 3: Keeping only active-cutting rows (Machining_Process contains 'Layer') ...")
active = data[data["Machining_Process"].str.contains("Layer", na=False)].copy()
print(f"-> Kept {active.shape[0]} of {data.shape[0]} rows\n")


# =========================================================================
# STEP 4: BUILD THE FEATURE TABLE (X) AND TARGET (y)
# =========================================================================
# X = everything the model is allowed to look at (the sensor readings)
# y = what we want it to predict (1 = worn, 0 = unworn)
# groups = which experiment each row belongs to (used ONLY for splitting,
#          never given to the model as a feature - that would be cheating,
#          since "experiment number" isn't something you'd know in real use)
non_feature_cols = ["y", "No", "Machining_Process"]
feature_cols = [c for c in active.select_dtypes(include=[np.number]).columns
                 if c not in non_feature_cols]

X = active[feature_cols]
y = active["y"]
groups = active["No"]

print(f"STEP 4: Using {len(feature_cols)} sensor columns as features.\n")


# =========================================================================
# STEP 5: EVALUATE HONESTLY WITH GROUP-BASED CROSS-VALIDATION
# =========================================================================
# GroupKFold splits the 18 experiments into 6 folds of 3 experiments each.
# In each round, it trains on 15 experiments and tests on the other 3 -
# and crucially, ALL rows from a test experiment are held out together,
# so the model never gets a sneak peek at an experiment it's being tested
# on. This gives us a realistic estimate of how the model would do on a
# brand new, never-before-seen machining run.
print("STEP 5: Cross-validating with GroupKFold (grouped by experiment) ...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
gkf = GroupKFold(n_splits=6)

cv_predictions = cross_val_predict(model, X, y, cv=gkf, groups=groups)

acc = accuracy_score(y, cv_predictions)
baseline = max(y.mean(), 1 - y.mean())  # accuracy from just guessing the majority class
cm = confusion_matrix(y, cv_predictions)

print(f"\n--- CROSS-VALIDATED RESULTS (the honest, leakage-free numbers) ---")
print(f"Accuracy:                {acc:.1%}")
print(f"'Always guess majority':  {baseline:.1%}  <- our model needs to beat this to be useful")
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(f"                 pred unworn   pred worn")
print(f"actual unworn        {cm[0][0]:>6}       {cm[0][1]:>6}")
print(f"actual worn          {cm[1][0]:>6}       {cm[1][1]:>6}")
print("\nFull report:")
print(classification_report(y, cv_predictions, target_names=["unworn", "worn"]))


# =========================================================================
# STEP 6: TRAIN A FINAL MODEL ON *ALL* THE DATA
# =========================================================================
# The cross-validation above is just for HONEST EVALUATION - it tells us
# how good the approach is. Once we're happy with that number, we train
# one more time on every single row we have, to get the actual model
# we'd use going forward (more training data = generally a better model).
print("STEP 6: Training the final model on all available data ...")
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(15)
print("\nTop 15 sensor features the model relied on most:")
print(top_features.to_string())


# =========================================================================
# STEP 7: SAVE PLOTS FOR YOUR REPORT
# =========================================================================
print("\nSTEP 7: Saving plots to the outputs/ folder ...")

# --- Confusion matrix plot ---
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["unworn", "worn"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["unworn", "worn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (accuracy = {acc:.1%})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                 color="white" if cm[i][j] > cm.max() / 2 else "black")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

# --- Feature importance plot ---
fig, ax = plt.subplots(figsize=(7, 5))
top_features.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_xlabel("Importance")
ax.set_title("Top 15 Most Important Sensor Features")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
plt.close(fig)

print(f"-> Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")
print(f"-> Saved: {OUTPUT_DIR / 'feature_importance.png'}")
print("\nDone! Open the two PNGs in outputs/ to drop straight into your report.")
