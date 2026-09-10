import os

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = "reports/experiment_results.csv"
FIGURES_DIR = "reports/figures"


print("=" * 70)
print("HOTEL BOOKING ML — MODEL COMPARISON")
print("=" * 70)


# Load experiment results
df = pd.read_csv(RESULTS_PATH)

print("\nAll model results:\n")

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)



# Ranking by F1 Score
f1_ranking = df.sort_values(
    by="f1",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 70)
print("RANKING BY F1 SCORE")
print("=" * 70)

for i, row in f1_ranking.iterrows():
    print(
        f"{i + 1}. {row['model']} "
        f"-> F1: {row['f1']:.4f}"
    )



# Ranking by ROC-AUC
roc_ranking = df.sort_values(
    by="roc_auc",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 70)
print("RANKING BY ROC-AUC")
print("=" * 70)

for i, row in roc_ranking.iterrows():
    print(
        f"{i + 1}. {row['model']} "
        f"-> ROC-AUC: {row['roc_auc']:.4f}"
    )


# Ranking by PR-AUC
pr_ranking = df.sort_values(
    by="pr_auc",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 70)
print("RANKING BY PR-AUC")
print("=" * 70)

for i, row in pr_ranking.iterrows():
    print(
        f"{i + 1}. {row['model']} "
        f"-> PR-AUC: {row['pr_auc']:.4f}"
    )



# Best models
best_f1 = df.loc[df["f1"].idxmax()]
best_roc = df.loc[df["roc_auc"].idxmax()]
best_pr = df.loc[df["pr_auc"].idxmax()]

print("\n" + "=" * 70)
print("BEST MODELS")
print("=" * 70)

print(
    f"\nBest F1 Score:\n"
    f"{best_f1['model']} -> {best_f1['f1']:.4f}"
)

print(
    f"\nBest ROC-AUC:\n"
    f"{best_roc['model']} -> {best_roc['roc_auc']:.4f}"
)

print(
    f"\nBest PR-AUC:\n"
    f"{best_pr['model']} -> {best_pr['pr_auc']:.4f}"
)


# Figure 1 — Classification Metrics
metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
]

x = range(len(df))
width = 0.18

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    values = df[metric]

    positions = [
        value + (i - 1.5) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric.upper(),
    )

plt.xticks(
    list(x),
    df["model"],
    rotation=20,
    ha="right",
)

plt.ylabel("Score")
plt.title("Classification Metrics by Model")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

path = os.path.join(
    FIGURES_DIR,
    "classification_metrics.png",
)

plt.savefig(path, dpi=300)
plt.show()
plt.close()

print(f"\nSaved: {path}")


# Figure 2 — ROC-AUC and PR-AUC
plt.figure(figsize=(10, 6))

x = range(len(df))
width = 0.35

plt.bar(
    [value - width / 2 for value in x],
    df["roc_auc"],
    width=width,
    label="ROC-AUC",
)

plt.bar(
    [value + width / 2 for value in x],
    df["pr_auc"],
    width=width,
    label="PR-AUC",
)

plt.xticks(
    list(x),
    df["model"],
    rotation=20,
    ha="right",
)

plt.ylabel("Score")
plt.title("ROC-AUC vs PR-AUC")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

path = os.path.join(
    FIGURES_DIR,
    "roc_auc_vs_pr_auc.png",
)

plt.savefig(path, dpi=300)
plt.show()
plt.close()

print(f"Saved: {path}")



# Figure 3 — Training Time
training_ranking = df.sort_values(
    by="training_time",
    ascending=True,
)

plt.figure(figsize=(10, 6))

plt.barh(
    training_ranking["model"],
    training_ranking["training_time"],
)

plt.xlabel("Training Time (seconds)")
plt.ylabel("Model")
plt.title("Model Training Time")

plt.tight_layout()

path = os.path.join(
    FIGURES_DIR,
    "training_time.png",
)

plt.savefig(path, dpi=300)
plt.show()
plt.close()

print(f"Saved: {path}")


# Fastest model
fastest = df.loc[
    df["training_time"].idxmin()
]

print(
    f"\nFastest Training:\n"
    f"{fastest['model']} -> "
    f"{fastest['training_time']:.3f} seconds"
)


print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)