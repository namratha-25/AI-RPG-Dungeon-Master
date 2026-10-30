# AI RPG Dungeon Master

An adaptive RPG engine where **ML models decide what happens** and a narrator only describes it.

```
player text -> Intent ML + Sentiment ML -> NPC behaviour ML -> world-state update
            -> player profiling (K-Means) + difficulty ML + quest recommender
            -> memory (short-term + long-term) -> narration
```

## Run it
```bash
pip install -r requirements.txt
python src/gen_data.py        # 1. create datasets   (data/*.csv)   - optional, CSVs are included
python src/train_nlp.py       # 2. intent + sentiment (models/, results/)
python src/train_models.py    # 3. NPC, clustering, difficulty, quests
streamlit run app.py          # 4. play
```
Models are already trained in `models/`; you can go straight to step 4.

## Files
| Path | What it is |
|---|---|
| `src/gen_data.py` | Synthetic datasets (intent, sentiment, NPC, player profile, quests) |
| `src/train_nlp.py` | Compares 7 algorithms for intent and sentiment (TF-IDF features) |
| `src/train_models.py` | NPC behaviour, K-Means/DBSCAN/Hierarchical, difficulty, quest recommender |
| `src/engine.py` | Game engine: ML chain, world state, NPC relationships, memory, narration |
| `app.py` | Streamlit UI (Play tab + Models & evaluation tab) |
| `results/` | Metric tables + confusion matrices + cluster plot (use these in slides) |

## Deployed model per task (chosen by weighted F1 on a held-out 20%)
| Task | Deployed | Compared against |
|---|---|---|
| Player intent | Logistic Regression | Naive Bayes, SVM, kNN, Decision Tree, Random Forest, XGBoost |
| Sentiment | Logistic Regression | same 6 |
| NPC behaviour | Gradient Boosting | Decision Tree, Random Forest, XGBoost |
| Difficulty | Logistic Regression | Random Forest, XGBoost |
| Player archetypes | K-Means (k=4 by silhouette) | DBSCAN, Hierarchical |
| Quest recommendation | Random Forest | content-based baseline, random baseline |

See `results/*.csv` for the exact numbers from your run.

## Honest notes (say these in the viva)
* All data is **synthetic**, generated from templates plus noise. High scores show the pipeline
  works, not real-world accuracy. Intent scores tie across several models for this reason.
* Quest RF and the content-based baseline perform about the same; both clearly beat random.
* Narration is template-based. The narrator never changes state, matching the deck's rule.
  `DungeonMaster.narrate()` is the single place to plug in an LLM later.
* Long-term memory uses TF-IDF similarity retrieval (not Sentence Transformers / a vector DB).
* Not built (future work): LSTM/BERT, Q-learning/DQN, SHAP, collaborative filtering.
