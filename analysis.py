"""
What makes a song a hit?  —  EDA + hit prediction on a music dataset.

Reads songs.csv (generate it first with `python generate_dataset.py`), explores
the audio features, and trains classifiers to predict whether a track is a "hit"
(popularity >= 70). All figures are saved to ./figures.

Run:  python analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cbd2dd", "axes.grid": True, "grid.color": "#eef1f5",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.autolayout": True,
})
HIT, MISS, ACCENT = "#1db954", "#b9c2d2", "#3753e8"   # Spotify green / grey / blue
AUDIO = ["danceability", "energy", "valence", "acousticness", "speechiness", "tempo", "loudness", "duration_min"]


def plot_hits_by_genre(df):
    rate = df.groupby("genre")["hit"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(rate.index, rate.values, color=HIT)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Hit rate by genre"); ax.set_ylabel("% of tracks that are hits")
    ax.set_ylim(0, max(rate.values) + 12)
    fig.savefig(f"{FIG}/hits_by_genre.png", dpi=130); plt.close(fig)
    return rate.round(1).to_dict()


def plot_feature_distributions(df):
    feats = ["danceability", "energy", "valence", "acousticness"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, f in zip(axes.ravel(), feats):
        ax.hist(df[df.hit == 1][f], bins=25, alpha=0.7, color=HIT, label="Hit", edgecolor="white", linewidth=0.4)
        ax.hist(df[df.hit == 0][f], bins=25, alpha=0.6, color=MISS, label="Not a hit", edgecolor="white", linewidth=0.4)
        ax.set_title(f); ax.legend(fontsize=9)
    fig.suptitle("Audio features: hits vs the rest", fontsize=14, fontweight="bold")
    fig.savefig(f"{FIG}/feature_distributions.png", dpi=130); plt.close(fig)


def plot_correlation(df):
    cols = AUDIO + ["popularity"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(corr.iloc[i, j]) > 0.55 else "#33415c")
    ax.set_title("Feature correlations (incl. popularity)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(f"{FIG}/correlation_heatmap.png", dpi=130); plt.close(fig)
    return corr["popularity"].drop("popularity").round(2).sort_values(ascending=False).to_dict()


def plot_popularity_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, f in zip(axes, ["danceability", "energy"]):
        for label, color, name in [(0, MISS, "Not a hit"), (1, HIT, "Hit")]:
            sub = df[df.hit == label]
            ax.scatter(sub[f], sub["popularity"], s=12, alpha=0.55, color=color, label=name, edgecolors="none")
        m, b = np.polyfit(df[f], df["popularity"], 1)
        xs = np.linspace(df[f].min(), df[f].max(), 50)
        ax.plot(xs, m * xs + b, color=ACCENT, lw=2.2, label="trend")
        ax.set_xlabel(f); ax.set_ylabel("popularity"); ax.set_title(f"Popularity vs {f}")
        ax.legend(fontsize=9, loc="upper left")
    fig.savefig(f"{FIG}/popularity_vs_features.png", dpi=130); plt.close(fig)


def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Greens"); labels = ["Not hit", "Hit"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title); ax.grid(False)
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#1e293b", fontsize=15, fontweight="bold")
    fig.savefig(f"{FIG}/confusion_matrix.png", dpi=130); plt.close(fig)


def plot_roc(y, p_lr, p_rf, a_lr, a_rf):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p, a, name, c in [(p_lr, a_lr, "Logistic Regression", ACCENT), (p_rf, a_rf, "Random Forest", HIT)]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9aa6c4", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — predicting a hit"); ax.legend(loc="lower right", fontsize=10)
    fig.savefig(f"{FIG}/roc_curve.png", dpi=130); plt.close(fig)


def plot_importance(model, names):
    imp = pd.Series(model.feature_importances_, index=names).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(imp.index, imp.values, color=ACCENT); ax.set_title("What drives a hit? (Random Forest importance)")
    ax.grid(False)
    fig.savefig(f"{FIG}/feature_importance.png", dpi=130); plt.close(fig)
    return imp.round(3).iloc[::-1].to_dict()


def evaluate(name, y, pred, proba):
    return {"model": name, "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def main():
    df = pd.read_csv("songs.csv")
    print(f"Loaded {len(df)} tracks · {df['hit'].mean()*100:.1f}% are hits")

    genre_rate = plot_hits_by_genre(df)
    plot_feature_distributions(df)
    pop_corr = plot_correlation(df)
    plot_popularity_scatter(df)
    print("\nCorrelation of features with popularity:", pop_corr)

    # features: audio + one-hot genre
    X = pd.concat([df[AUDIO], pd.get_dummies(df["genre"], prefix="genre")], axis=1)
    y = df["hit"].values
    feat_names = list(X.columns)

    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)

    lr = LogisticRegression(max_iter=5000).fit(scaler.transform(Xtr), ytr)
    lr_pred = lr.predict(scaler.transform(Xte)); lr_proba = lr.predict_proba(scaler.transform(Xte))[:, 1]

    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
    rf_pred = rf.predict(Xte); rf_proba = rf.predict_proba(Xte)[:, 1]

    results = [evaluate("Logistic Regression", yte, lr_pred, lr_proba),
               evaluate("Random Forest", yte, rf_pred, rf_proba)]

    plot_roc(yte, lr_proba, rf_proba, results[0]["roc_auc"], results[1]["roc_auc"])
    best = int(np.argmax([r["roc_auc"] for r in results]))
    best_pred = lr_pred if best == 0 else rf_pred
    plot_confusion(confusion_matrix(yte, best_pred), f"Confusion — {results[best]['model']}")
    top = plot_importance(rf, feat_names)

    print("\n=== Predicting a hit (25% hold-out) ===")
    hdr = f"{'Model':22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'ROC-AUC':>9}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['model']:22}{r['accuracy']:8.3f}{r['precision']:8.3f}{r['recall']:8.3f}{r['f1']:8.3f}{r['roc_auc']:9.3f}")
    cv = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42), X.values, y, cv=5).mean()
    print(f"\nRandom Forest 5-fold CV accuracy: {cv:.3f}")
    print("\nTop hit-driving features:", list(top.items())[:6])
    print("\nClassification report (best model):")
    print(classification_report(yte, best_pred, target_names=["Not hit", "Hit"]))

    pd.DataFrame(results).set_index("model").round(4).to_csv("results.csv")
    print("Saved figures -> ./figures, metrics -> results.csv")
    return results, genre_rate, pop_corr, top, cv


if __name__ == "__main__":
    main()
