"""Step 3/4 - NPC behaviour, player clustering, difficulty prediction, quest recommendation.
Run:  python src/train_models.py
Outputs: models/*.joblib, results/*.csv, results/*.png
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold,
                                     GroupShuffleSplit)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             silhouette_score, davies_bouldin_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA, MODELS, RES = (os.path.join(ROOT, d) for d in ("data", "models", "results"))
os.makedirs(MODELS, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def metrics_row(name, yte, p):
    return {"model": name,
            "accuracy": accuracy_score(yte, p),
            "precision": precision_score(yte, p, average="weighted", zero_division=0),
            "recall": recall_score(yte, p, average="weighted", zero_division=0),
            "f1": f1_score(yte, p, average="weighted")}


def compare_classifiers(task, df, target, cat_cols, num_cols, cands):
    """Train every candidate, print table, save the winner (refit on all data)."""
    print(f"\n===== {task.upper()} =====")
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    X = df[cat_cols + num_cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def make(clf):
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                  ("num", "passthrough", num_cols)])
        return Pipeline([("prep", prep), ("clf", clf)])

    rows, preds = [], {}
    for name, factory in cands.items():
        pipe = make(factory())
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        cv = cross_val_score(make(factory()), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring="f1_weighted").mean()
        r = metrics_row(name, yte, p)
        r["cv_f1_5fold"] = cv
        rows.append(r)
        preds[name] = p
        print(f"  {name:<20} acc={r['accuracy']:.3f}  f1={r['f1']:.3f}  cv_f1={cv:.3f}")
    res = pd.DataFrame(rows).sort_values("f1", ascending=False).round(4)
    res.to_csv(f"{RES}/{task}_results.csv", index=False)
    best = res.iloc[0]["model"]
    print(f"  >> BEST for {task}: {best}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[best]), display_labels=le.classes_).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{task}: {best}")
    fig.tight_layout()
    fig.savefig(f"{RES}/{task}_confusion.png", dpi=130)
    plt.close(fig)

    final = make(cands[best]()).fit(X, y)
    joblib.dump({"model": final, "le": le, "name": best, "cat": cat_cols, "num": num_cols},
                f"{MODELS}/{task}_model.joblib")
    return res


def xgb():
    return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric="mlogloss",
                         random_state=42)


# --------------------------------------------------------------------------- NPC BEHAVIOUR
npc = pd.read_csv(f"{DATA}/03_npc_behavior_dataset.csv")
compare_classifiers(
    "npc_behavior", npc, "npc_reaction",
    cat_cols=["npc_type", "player_intent", "player_sentiment"],
    num_cols=["trust", "fear", "anger", "loyalty", "greed"],
    cands={"Decision Tree": lambda: DecisionTreeClassifier(max_depth=10, random_state=42),
           "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
           "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
           "XGBoost": xgb})

# --------------------------------------------------------------------------- DIFFICULTY
prof = pd.read_csv(f"{DATA}/04_player_profile_dataset.csv")
order = {"EASY": 0, "MEDIUM": 1, "HARD": 2}
compare_classifiers(
    "difficulty", prof, "recommended_difficulty", cat_cols=[],
    num_cols=["player_level", "deaths", "success_rate"],
    cands={"Logistic Regression": lambda: LogisticRegression(max_iter=3000),
           "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
           "XGBoost": xgb})

# --------------------------------------------------------------------------- PLAYER CLUSTERING
print("\n===== PLAYER ARCHETYPE CLUSTERING =====")
FEATS = ["combat_rate", "stealth_rate", "exploration_rate", "dialogue_rate"]
Z = prof[FEATS].values
rows = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    rows.append({"k": k, "silhouette": silhouette_score(Z, km.labels_),
                 "davies_bouldin": davies_bouldin_score(Z, km.labels_)})
kdf = pd.DataFrame(rows).round(4)
kdf.to_csv(f"{RES}/kmeans_k_selection.csv", index=False)
print(kdf.to_string(index=False))
best_k = int(kdf.sort_values("silhouette", ascending=False).iloc[0]["k"])
print(f"  >> chosen k = {best_k}")

km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Z)
db = DBSCAN(eps=0.12, min_samples=8).fit(Z)
ag = AgglomerativeClustering(n_clusters=best_k).fit(Z)
comp = []
for name, lab in [("K-Means", km.labels_), ("Hierarchical", ag.labels_), ("DBSCAN", db.labels_)]:
    mask = lab != -1                                 # DBSCAN noise points excluded from scoring
    n_cl = len(set(lab[mask]))
    if n_cl >= 2:
        comp.append({"algorithm": name, "clusters": n_cl,
                     "silhouette": silhouette_score(Z[mask], lab[mask]),
                     "davies_bouldin": davies_bouldin_score(Z[mask], lab[mask])})
    else:
        comp.append({"algorithm": name, "clusters": n_cl, "silhouette": np.nan, "davies_bouldin": np.nan})
cdf = pd.DataFrame(comp).round(4)
cdf.to_csv(f"{RES}/clustering_comparison.csv", index=False)
print(cdf.to_string(index=False))

# name each cluster by its dominant behaviour
ARCH_NAMES = ["Warrior", "Assassin", "Explorer", "Diplomat"]
names, used = {}, {}
for c, center in enumerate(km.cluster_centers_):
    base = ARCH_NAMES[int(np.argmax(center))]
    used[base] = used.get(base, 0) + 1
    names[c] = base if used[base] == 1 else f"{base} {used[base]}"
print("  cluster names:", names)
joblib.dump({"kmeans": km, "names": names, "features": FEATS}, f"{MODELS}/archetype_kmeans.joblib")

pc = PCA(n_components=2).fit_transform(Z)
fig, ax = plt.subplots(figsize=(6.5, 5))
for c in range(best_k):
    ax.scatter(pc[km.labels_ == c, 0], pc[km.labels_ == c, 1], s=12, label=names[c], alpha=0.7)
ax.legend()
ax.set_title(f"Player archetypes (K-Means, k={best_k})")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
fig.tight_layout()
fig.savefig(f"{RES}/archetype_clusters.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- QUEST RECOMMENDATION
print("\n===== QUEST RECOMMENDATION =====")
q = pd.read_csv(f"{DATA}/05_quest_recommendation_dataset.csv")
PREF = ["combat_pref", "exploration_pref", "dialogue_pref", "stealth_pref", "rare_weapon_pref"]
QC = ["quest_name", "quest_type", "quest_difficulty"]
TYPE2PREF = {"COMBAT": "combat_pref", "EXPLORATION": "exploration_pref", "DIALOGUE": "dialogue_pref",
             "STEALTH": "stealth_pref", "TREASURE": "rare_weapon_pref"}
q["matched_pref"] = [r[TYPE2PREF[r["quest_type"]]] for _, r in q.iterrows()]   # engineered feature
FEAT = PREF + ["matched_pref"] + QC


def quest_pipe():
    prep = ColumnTransformer([
        ("num", "passthrough", PREF + ["matched_pref"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["quest_name", "quest_type"]),
        ("diff", OrdinalEncoder(categories=[["EASY", "MEDIUM", "HARD"]]), ["quest_difficulty"])])
    return Pipeline([("prep", prep), ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42))])


def precision_at_k(df, col, k=3):
    return np.mean([g.sort_values(col, ascending=False).head(k)["liked_quest"].mean()
                    for _, g in df.groupby("player_id")])


# evaluate on UNSEEN players, averaged over 5 random player-splits (single splits are noisy)
scores = {"Random Forest (ML)": [], "Content-based (pref match)": [], "Random baseline": []}
accs = []
for seed in range(5):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(q, groups=q["player_id"]))
    qtr, qte = q.iloc[tr], q.iloc[te].copy()
    m = quest_pipe().fit(qtr[FEAT], qtr["liked_quest"])
    qte["rf"] = m.predict_proba(qte[FEAT])[:, 1]
    qte["cb"] = qte["matched_pref"]
    qte["rand"] = np.random.default_rng(seed).random(len(qte))
    accs.append(accuracy_score(qte["liked_quest"], (qte["rf"] > 0.5).astype(int)))
    for name, col in zip(scores, ["rf", "cb", "rand"]):
        scores[name].append((precision_at_k(qte, col), roc_auc_score(qte["liked_quest"], qte[col])))
rec = pd.DataFrame([{"method": n, "precision@3": np.mean([x[0] for x in v]),
                     "roc_auc": np.mean([x[1] for x in v]),
                     "accuracy": np.mean(accs) if n.startswith("Random Forest") else np.nan}
                    for n, v in scores.items()]).round(4)
rec.to_csv(f"{RES}/quest_results.csv", index=False)
print(rec.to_string(index=False))

final_q = quest_pipe().fit(q[FEAT], q["liked_quest"])
joblib.dump({"model": final_q, "quests": q[QC].drop_duplicates().reset_index(drop=True),
             "pref_cols": PREF, "features": FEAT}, f"{MODELS}/quest_model.joblib")
print("\nAll models saved to models/, tables & plots to results/")
