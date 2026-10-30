# 🐉 AI RPG Dungeon Master

<h1 align="center">🎮 AI RPG Dungeon Master</h1>

<h3 align="center">⚔️ Every Choice Shapes Your World. Every Action Has Consequences. 🌍</h3>

<p align="center">
  An adaptive, Machine Learning-powered RPG where artificial intelligence controls NPC behaviour, understands player intentions, adapts difficulty, and recommends personalized quests.
</p>

<p align="center">
  <a href="https://ai-rpg-dungeon-master.streamlit.app">
    <img src="https://img.shields.io/badge/🎮_PLAY_NOW-Live_Demo-FF4B4B?style=for-the-badge" alt="Live Demo"/>
  </a>
  <a href="https://github.com/namratha-25/AI-RPG-Dungeon-Master">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikitlearn"/>
  <img src="https://img.shields.io/badge/XGBoost-ML-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Deployed-success?style=flat-square"/>
</p>

---

## 🌟 About the Project

**AI RPG Dungeon Master** is an intelligent, interactive role-playing game that combines Natural Language Processing (NLP), Machine Learning, and dynamic storytelling.

Unlike traditional RPGs that rely on predefined responses, this project uses trained ML models to understand player actions, predict NPC reactions, track relationships, and adapt the gaming experience.

Players can attack, negotiate, help, intimidate, explore, and make decisions that influence the game world.

🧠 **The central idea:** ML models decide what happens. The narrator only describes the outcome.

## 🚀 Experience the Game

<p align="center">
  <a href="https://ai-rpg-dungeon-master.streamlit.app">
    <img src="https://img.shields.io/badge/CLICK_HERE_TO_PLAY-🐉_ENTER_THE_DUNGEON-FF4B4B?style=for-the-badge" alt="Play Game"/>
  </a>
</p>

🌍 [Open AI RPG Dungeon Master](https://ai-rpg-dungeon-master.streamlit.app)

Explore the world, interact with characters, make decisions, and discover how the AI responds to your actions!

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 Player Intent Recognition | Understands player actions using NLP and Machine Learning. |
| 😊 Sentiment Analysis | Identifies positive, negative, and neutral player sentiment. |
| 🤖 Intelligent NPC Behaviour | Predicts NPC reactions based on player actions and relationships. |
| ❤️ Relationship Tracking | Maintains trust, fear, anger, and loyalty for NPCs. |
| 🌍 Dynamic World State | Updates reputation, town alerts, gold, and dragon activity. |
| 🎯 Player Profiling | Uses K-Means clustering to identify player behaviour archetypes. |
| ⚔️ Adaptive Difficulty | Predicts suitable game difficulty based on player performance. |
| 📜 Quest Recommendation | Recommends quests based on player preferences and difficulty. |
| 🧠 Game Memory | Maintains short-term and long-term interaction memory. |
| 📊 Model Evaluation | Displays model performance and evaluation visualizations. |
| 💾 Game Management | Provides Save, Load, and Reset functionality. |

---

## 🔄 How It Works

The game follows an ML-driven decision-making pipeline.

```text
          🎮 PLAYER INPUT
                 |
                 ▼
       🧠 INTENT RECOGNITION
                 |
                 ▼
        😊 SENTIMENT ANALYSIS
                 |
                 ▼
        🤖 NPC BEHAVIOUR MODEL
                 |
                 ▼
         🌍 WORLD STATE UPDATE
                 |
          ┌──────┴──────┐
          ▼             ▼
    🎯 PLAYER       ⚔️ DIFFICULTY
    PROFILING        PREDICTION
          |             |
          └──────┬──────┘
                 ▼
        📜 QUEST RECOMMENDER
                 |
                 ▼
          🧠 GAME MEMORY
                 |
                 ▼
        📖 DYNAMIC NARRATION
                 |
                 ▼
          🎮 NEXT TURN
```

### 🧩 The AI Decision Pipeline

1. **Player Input:** The player enters a natural-language action.
2. **Intent Recognition:** The NLP model identifies the player's intention.
3. **Sentiment Analysis:** The sentiment model evaluates the emotional tone.
4. **NPC Behaviour:** The NPC model predicts how the character reacts.
5. **World Update:** Reputation, relationships, alerts, and other game variables change.
6. **Player Profiling:** K-Means identifies the player's behaviour pattern.
7. **Difficulty Prediction:** The game evaluates player performance and recommends difficulty.
8. **Quest Recommendation:** The recommendation model selects suitable quests.
9. **Memory and Narration:** Previous interactions are retrieved and the narrator describes the result.

---

## 🤖 Machine Learning Models

The project uses different ML techniques for different game functionalities.

| Task | Selected Algorithm | Purpose |
|---|---|---|
| 🎯 Player Intent | Logistic Regression | Classifies player actions |
| 😊 Sentiment Analysis | Logistic Regression | Identifies player sentiment |
| 🤖 NPC Behaviour | Gradient Boosting | Predicts NPC reactions |
| ⚔️ Difficulty Prediction | Logistic Regression | Recommends game difficulty |
| 🧩 Player Profiling | K-Means Clustering | Identifies player archetypes |
| 📜 Quest Recommendation | Random Forest | Recommends personalized quests |

### 🔬 Model Evaluation

The project compares multiple algorithms using evaluation metrics such as weighted F1-score, confusion matrices, and clustering analysis.

The deployed models were selected based on their performance on a held-out 20% test split.

Evaluation visualizations and result files are included in the repository.

---

## 🛠️ Tech Stack

**Programming Language**
- Python

**Machine Learning & Data Science**
- Scikit-learn
- XGBoost
- NumPy
- Pandas
- Joblib

**Natural Language Processing**
- TF-IDF Vectorization
- Text Classification
- Cosine Similarity

**Frontend & Deployment**
- Streamlit
- Streamlit Community Cloud

**Development Tools**
- Git
- GitHub
- VS Code

---

## 📂 Project Structure

```text
AI-RPG-Dungeon-Master/
│
├── 📄 app.py
├── ⚙️ engine.py
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📊 01_player_intent_dataset.csv
├── 📊 02_sentiment_dataset.csv
├── 📊 03_npc_behavior_dataset.csv
├── 📊 04_player_profile_dataset.csv
├── 📊 05_quest_recommendation_dataset.csv
│
├── 🤖 intent_model.joblib
├── 🤖 sentiment_model.joblib
├── 🤖 npc_behavior_model.joblib
├── 🤖 difficulty_model.joblib
├── 🤖 archetype_kmeans.joblib
├── 🤖 quest_model.joblib
│
├── 📈 intent_confusion.png
├── 📈 sentiment_confusion.png
├── 📈 npc_behavior_confusion.png
├── 📈 difficulty_confusion.png
├── 📈 archetype_clusters.png
│
├── 🧪 gen_data.py
├── 🧪 train_nlp.py
├── 🧪 train_models.py
│
└── 📊 Evaluation CSV files
```

---

## 💻 Installation & Local Setup

Want to run the project on your own machine?

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/namratha-25/AI-RPG-Dungeon-Master.git
```

### 2️⃣ Navigate to the Project

```bash
cd AI-RPG-Dungeon-Master
```

### 3️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Launch the Application

```bash
streamlit run app.py
```

🎉 The application will open in your browser!

**Note:** Pre-trained ML model files are included in the repository. You can run the application directly without retraining the models.

---

## 🎮 Try These Player Actions

Not sure what to type? Start with these examples!

| Player Action | Expected Intent |
|---|---|
| ⚔️ I attack the guard | Attack |
| 🤝 Please help me, friend | Help |
| 💰 I want to negotiate a trade | Negotiate |
| 🕵️ I secretly steal the gold | Steal |
| 🌲 I explore the mysterious forest | Explore |
| 😈 I threaten the merchant | Intimidate |
| 💬 I persuade the captain to trust me | Persuade |

*Actual NPC reactions depend on the trained models and current game state.*

---

## 📊 Models & Evaluation

The application includes a dedicated **Models & Evaluation** section.

It provides access to:
- Model comparison results.
- Confusion matrices.
- Clustering visualizations.
- Performance evaluation results.

These visualizations help understand how different ML algorithms contribute to the game.

---

## 🔮 Future Enhancements

- 🗣️ Integrate LLM-powered conversational storytelling.
- 🧠 Implement advanced memory using vector databases.
- 🎨 Add an immersive RPG interface with animations.
- 🗺️ Introduce interactive maps and world exploration.
- 🎙️ Add voice-based player interaction.
- 🕹️ Explore reinforcement learning for adaptive NPC strategies.
- ☁️ Implement persistent cloud-based game storage.

---

## ⚠️ Project Limitations

- The datasets are synthetically generated using templates and noise.
- High evaluation scores demonstrate the working ML pipeline, not necessarily real-world generalization.
- Narration is currently template-based rather than LLM-generated.
- Long-term memory uses TF-IDF and cosine similarity rather than a dedicated vector database.
- Reinforcement learning and advanced transformer-based NLP are potential future enhancements.

---

## 👩‍💻 Developed With Passion

Built as a Machine Learning and Artificial Intelligence project to explore how intelligent systems can create more adaptive and engaging gaming experiences.

**Technologies. Intelligence. Imagination. One Dungeon. Endless Possibilities.** 🐉

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>

<p align="center">
  🎮 <a href="https://ai-rpg-dungeon-master.streamlit.app"><b>PLAY THE GAME</b></a> |
  💻 <a href="https://github.com/namratha-25/AI-RPG-Dungeon-Master"><b>EXPLORE THE CODE</b></a>
</p>

<p align="center">
  <b>Made with ❤️, Python 🐍 and Machine Learning 🧠</b>
</p>
