"""Step 2 - train & compare models for Player Intent and Sentiment.
Run:  python src/train_nlp.py
Outputs: models/intent_model.joblib, models/sentiment_model.joblib,
         results/intent_results.csv, results/sentiment_results.csv, results/*_confusion.png
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA, MODELS, RES = (os.path.join(ROOT, d) for d in ("data", "models", "results"))
os.makedirs(MODELS, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def candidates():
    # fresh instances each call (so intent and sentiment don't share fitted state)
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Naive Bayes": MultinomialNB(),
        "SVM (linear)": CalibratedClassifierCV(LinearSVC(), cv=3),   # calibrated -> gives probabilities
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 eval_metric="mlogloss", random_state=42),
    }


def run_task(name, df, text_col, target_col):
    print(f"\n===== {name.upper()} =====")
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X = df[text_col].fillna("").astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rows, fitted = [], {}
    for mname, clf in candidates().items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring="f1_weighted").mean()
        rows.append({"model": mname,
                     "accuracy": accuracy_score(yte, p),
                     "precision": precision_score(yte, p, average="weighted", zero_division=0),
                     "recall": recall_score(yte, p, average="weighted", zero_division=0),
                     "f1": f1_score(yte, p, average="weighted"),
                     "cv_f1_5fold": cv})
        fitted[mname] = (pipe, p)
        print(f"  {mname:<20} acc={rows[-1]['accuracy']:.3f}  f1={rows[-1]['f1']:.3f}  cv_f1={cv:.3f}")

    res = pd.DataFrame(rows).sort_values("f1", ascending=False).round(4)
    res.to_csv(f"{RES}/{name}_results.csv", index=False)
    best = res.iloc[0]["model"]
    print(f"  >> BEST for {name}: {best}")

    # confusion matrix of the winner
    pipe, pred = fitted[best]
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=le.classes_).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{name}: {best}")
    fig.tight_layout()
    fig.savefig(f"{RES}/{name}_confusion.png", dpi=130)
    plt.close(fig)

    # refit winner on ALL data, save with label encoder
    final = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                      ("clf", candidates()[best])]).fit(X, y)
    joblib.dump({"model": final, "le": le, "name": best}, f"{MODELS}/{name}_model.joblib")


if __name__ == "__main__":
    run_task("intent", pd.read_csv(f"{DATA}/01_player_intent_dataset.csv"), "player_text", "intent")
    run_task("sentiment", pd.read_csv(f"{DATA}/02_sentiment_dataset.csv"), "player_message", "sentiment")
    print("\nSaved models to models/ and tables/plots to results/")
