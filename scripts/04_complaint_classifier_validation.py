"""
04_complaint_classifier_validation.py
---------------------------------------------------------------------
Core "model / GenAI output validation" component.

Trains a baseline text classifier to predict the root-cause category
(Issue) from the free-text complaint narrative, then runs it through
a structured VALIDATION framework: accuracy, per-class precision/
recall, a confusion matrix, and a written error-pattern breakdown.

This mirrors two specific JD lines almost exactly:
  - "Assist with validation efforts on model and Gen AI work to
     enhance our complaints insights"
  - "Experience with reviewing output of Gen AI tools or validating
     model output" (preferred qualification)

An OPTIONAL section at the bottom shows how to swap the classical
model for a real LLM call (e.g. Claude or GPT) so the exact same
validation harness can audit GenAI output instead of / alongside a
classical model -- see the README for how to wire in an API key.
---------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

DATA_PATH = "/home/claude/complaints_project/data/sample_complaints.csv"
OUT_DIR = "/home/claude/complaints_project/outputs"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["consumer_complaint_narrative", "issue"])

X = df["consumer_complaint_narrative"]
y = df["issue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 1. Baseline model: TF-IDF + Logistic Regression
#    (stand-in for "the model/GenAI system" whose output we validate)
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

# ---------------------------------------------------------------------
# 2. VALIDATION -- this is the deliverable, not the model itself
# ---------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

print(f"Overall accuracy on held-out validation set: {acc:.1%}\n")
print("=== Per-class precision / recall / F1 ===")
print(report)

# Save a full validation report to file, the kind of artifact you'd
# actually hand to a business/compliance stakeholder.
with open(f"{OUT_DIR}/model_validation_report.txt", "w") as f:
    f.write("ROOT-CAUSE CLASSIFIER -- MODEL VALIDATION REPORT\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Validation set size: {len(y_test)}\n")
    f.write(f"Overall accuracy: {acc:.1%}\n\n")
    f.write("Per-class precision / recall / F1:\n")
    f.write(report)
    f.write("\n\nERROR ANALYSIS\n" + "-" * 55 + "\n")

    # ---------------------------------------------------------------
    # 3. Error-pattern breakdown -- WHERE and WHY the model is wrong,
    #    not just how often. This is what turns "I ran a model" into
    #    "I validated a model", which is the actual JD ask.
    # ---------------------------------------------------------------
    errors = pd.DataFrame({"true": y_test.values, "pred": y_pred})
    errors = errors[errors["true"] != errors["pred"]]
    confusion_pairs = (
        errors.groupby(["true", "pred"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    f.write("Top confused label pairs (true -> predicted):\n")
    for _, row in confusion_pairs.iterrows():
        f.write(f"  {row['true']!r} -> {row['pred']!r}: {row['count']} cases\n")

    f.write(f"\nTotal misclassifications: {len(errors)} / {len(y_test)} "
             f"({len(errors)/len(y_test):.1%})\n")
    f.write(
        "\nInterpretation: misclassifications concentrate between labels\n"
        "that share overlapping vocabulary (e.g. two Issues under the same\n"
        "Product). In a production setting this is exactly the kind of\n"
        "pattern a human validator would flag back to the model/prompt\n"
        "owner -- e.g. recommending additional distinguishing features,\n"
        "more training examples for the confused pair, or a human-in-the-\n"
        "loop review step for low-confidence predictions.\n"
    )

print(f"Full validation report written to {OUT_DIR}/model_validation_report.txt")

# ---------------------------------------------------------------------
# 4. Confusion matrix heatmap
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")
short_labels = [l if len(l) < 28 else l[:25] + "..." for l in labels]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
ax.set_yticklabels(short_labels, fontsize=7)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Root-Cause Classifier -- Confusion Matrix (acc={acc:.1%})")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_confusion_matrix.png", dpi=120)
plt.close()

print(f"Confusion matrix chart written to {OUT_DIR}/04_confusion_matrix.png")

# ---------------------------------------------------------------------
# OPTIONAL -- swap in a real LLM instead of the classical model
# ---------------------------------------------------------------------
# To validate an actual GenAI system instead of / alongside the
# baseline above, replace the `clf.predict` call with something like:
#
#   import anthropic
#   client = anthropic.Anthropic(api_key="YOUR_KEY")
#   def llm_classify(narrative, allowed_labels):
#       msg = client.messages.create(
#           model="claude-sonnet-4-6",
#           max_tokens=30,
#           messages=[{"role": "user", "content":
#               f"Classify this complaint into exactly one of these "
#               f"categories: {allowed_labels}.\n\nComplaint: {narrative}\n\n"
#               f"Respond with only the category name."}]
#       )
#       return msg.content[0].text.strip()
#
#   y_pred_llm = [llm_classify(n, labels) for n in X_test]
#
# Then run the exact same validation block above (accuracy_score,
# classification_report, confusion_matrix, error analysis) on
# y_pred_llm instead of y_pred. Comparing the LLM's error pattern to
# the classical baseline's is itself a strong insight to present --
# it's literally "reviewing output of Gen AI tools ... validating
# model output," the preferred qualification named in the JD.
