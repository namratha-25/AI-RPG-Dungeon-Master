"""Step 1 - generate all synthetic datasets for the AI RPG Dungeon Master.
Run:  python src/gen_data.py
Creates 5 CSVs in ./data (column layout matches the original Tkinter app).
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)


def pick(lst):
    return lst[rng.integers(len(lst))]


# ---------------------------------------------------------------- 01 INTENT
# columns: player_text | location | player_level | health_pct | intent   (text first, target last)
TARGETS = ["the guard", "the dragon", "the merchant", "the goblin", "the king",
           "the innkeeper", "the bandit", "the wizard", "the villager", "the priest"]
OBJECTS = ["the chest", "the door", "the cave", "the ruins", "the tower", "the forest path"]
FILL = ["quickly", "carefully", "now", "slowly", "with my sword", "in the dark", "again", "without a word"]
INTENT_T = {
    "attack":     ["Attack {t}!", "Kill {t}", "Strike {t} down", "Fight {t} to the death", "Smash {t} with my axe", "I attack {t}", "I draw my sword and strike {t}", "I charge at {t} and swing", "Let me slash {t} down",
                   "I fire an arrow at {t}", "I cast a fireball at {t}", "I hit {t} hard"],
    "negotiate":  ["Make a deal with {t}", "Let us trade, {t}", "What will you pay me, {t}?", "Name your price, {t}", "I offer {t} a fair deal", "Let us make a bargain, {t}", "I propose a trade with {t}",
                   "How much for that, {t}?", "I want to haggle with {t}", "Can we reach an agreement, {t}?"],
    "steal":      ["Steal from {t}", "Pickpocket {t} quietly", "Sneak and rob {t}", "Take the gold and run, unseen", "I sneak behind {t} and pick the pocket", "I try to steal the gold from {t}",
                   "I quietly lift the keys from {t}", "I rob {t} while no one looks", "I swipe the jewel and hide it"],
    "explore":    ["Search {o}", "Look around {o}", "Investigate the area", "Check inside {o}", "I search {o}", "I walk into {o} to look around", "I examine {o} closely",
                   "I investigate {o}", "I open {o} and look inside", "I explore beyond {o}"],
    "flee":       ["Run!", "Escape from {t}", "Retreat now, back away", "Get away from {t}", "I run away from {t}", "I retreat and escape through {o}", "I flee the scene",
                   "I back off and run for my life", "I get out of here, away from {t}"],
    "help":       ["Please help {t}", "Heal {t}, kind friend", "Save {t} from danger", "Aid {t} and share my supplies", "I heal {t} with a potion", "I help {t} carry the load", "I protect {t} from harm",
                   "I rescue {t}", "I offer {t} my food and aid", "I assist {t} with the wounds"],
    "persuade":   ["Convince {t} to trust me", "Please listen to reason, {t}", "Win over {t} with kind words", "Talk {t} into helping us", "I convince {t} to join my cause", "I talk {t} into letting us pass",
                   "I appeal to the honor of {t}", "I charm {t} with kind words", "I reason with {t} calmly"],
    "intimidate": ["Threaten {t}!", "Scare {t} into talking", "Hand it over or else, {t}", "Warn {t} that they will suffer", "I threaten {t} to hand it over", "I glare at {t} and growl a warning",
                   "I demand {t} obey me or suffer", "I scare {t} with my blade drawn", "I menace {t} until they talk"],
}
LOCS = ["tavern", "dungeon", "castle", "forest", "village", "mountain", "market"]
rows = []
for _ in range(1600):
    intent = pick(list(INTENT_T))
    tpl = pick(INTENT_T[intent])
    text = tpl.format(t=pick(TARGETS), o=pick(OBJECTS))
    if rng.random() < 0.5:
        text += " " + pick(FILL)
    if rng.random() < 0.08:  # noisy word swap -> keeps accuracy realistic
        text = text.replace("I ", pick(["I try to ", "I really ", "I "]), 1)
    label = intent
    if rng.random() < 0.04:  # 4% label noise
        label = pick(list(INTENT_T))
    rows.append((text, pick(LOCS), int(rng.integers(1, 31)), int(rng.integers(5, 101)), label))
pd.DataFrame(rows, columns=["player_text", "location", "player_level", "health_pct", "intent"]) \
    .to_csv(f"{OUT}/01_player_intent_dataset.csv", index=False)

# ---------------------------------------------------------------- 02 SENTIMENT
POS = ["please", "kindly", "thank you", "friend", "gladly", "honored", "wonderful", "generous", "respect", "gratitude"]
NEG = ["damn", "filthy", "coward", "die", "hate", "worthless", "pathetic", "fool", "curse you", "rotten"]
NEU = ["the road is long", "I need information", "tell me about the town", "where is the gate",
       "what is the price", "I am traveling north", "show me the map", "the weather is cold"]
rows = []
for _ in range(1500):
    s = pick(["positive", "negative", "neutral"])
    pool = {"positive": POS, "negative": NEG, "neutral": []}[s]
    parts = [pick(NEU)] + [pick(pool) for _ in range(rng.integers(0, 3) if s != "neutral" else 0)]
    if s != "neutral" and rng.random() < 0.15:           # mixed-signal noise
        parts.append(pick(POS if s == "negative" else NEG))
    rng.shuffle(parts)
    label = s if rng.random() > 0.04 else pick(["positive", "negative", "neutral"])
    rows.append((" ".join(parts), label))
pd.DataFrame(rows, columns=["player_message", "sentiment"]).to_csv(f"{OUT}/02_sentiment_dataset.csv", index=False)

# ---------------------------------------------------------------- 03 NPC BEHAVIOUR
NPC_TYPES = ["guard", "merchant", "priest", "bandit", "noble", "villager"]
INTENTS = list(INTENT_T)
SENTS = ["positive", "negative", "neutral"]
rows = []
for _ in range(2000):
    t = {k: round(float(rng.random()), 2) for k in ["trust", "fear", "anger", "loyalty", "greed"]}
    npc, intent, sent = pick(NPC_TYPES), pick(INTENTS), pick(SENTS)
    sc = {"help": 0, "ignore": 0.3, "threaten": 0, "negotiate": 0, "attack": 0, "flee": 0}
    sentv = {"positive": 0.4, "neutral": 0, "negative": -0.4}[sent]
    if intent in ("help", "persuade"):
        sc["help"] += t["trust"] + sentv + 0.5; sc["negotiate"] += 0.3
    if intent == "negotiate":
        sc["negotiate"] += 0.8 + t["greed"] * 0.8; sc["help"] += t["trust"] * 0.3
    if intent in ("attack", "steal"):
        sc["attack"] += t["anger"] + 0.3; sc["flee"] += t["fear"] + 0.2; sc["threaten"] += t["anger"] * 0.5
    if intent == "intimidate":
        sc["flee"] += t["fear"] * 1.3; sc["threaten"] += t["anger"] * 1.2; sc["negotiate"] += t["greed"] * 0.4
    if intent in ("explore", "flee"):
        sc["ignore"] += 0.8
    if npc == "bandit":
        sc["attack"] += 0.3; sc["help"] -= 0.3
    if npc == "guard":
        sc["attack"] += 0.2 if intent in ("attack", "steal") else 0
    if npc == "merchant":
        sc["negotiate"] += 0.3
    if npc == "priest":
        sc["help"] += 0.2
    sc = {k: v + rng.normal(0, 0.15) for k, v in sc.items()}   # noise
    reaction = max(sc, key=sc.get)
    rel = float(np.clip((t["trust"] + t["loyalty"] - t["anger"] - t["fear"] * 0.3) * 60 + sentv * 40
                        + rng.normal(0, 8), -100, 100))
    rows.append((npc, *t.values(), intent, sent, round(rel, 1), reaction))
pd.DataFrame(rows, columns=["npc_type", "trust", "fear", "anger", "loyalty", "greed",
                            "player_intent", "player_sentiment", "relationship_score", "npc_reaction"]) \
    .to_csv(f"{OUT}/03_npc_behavior_dataset.csv", index=False)

# ---------------------------------------------------------------- 04 PLAYER PROFILE (clusters + difficulty)
ARCH = {  # combat, stealth, exploration, dialogue  (mean rates)
    "Warrior":  [0.60, 0.08, 0.17, 0.15], "Assassin": [0.15, 0.60, 0.15, 0.10],
    "Explorer": [0.12, 0.15, 0.60, 0.13], "Diplomat": [0.10, 0.10, 0.15, 0.65]}
rows = []
for _ in range(800):
    a = pick(list(ARCH))
    r = rng.dirichlet(np.array(ARCH[a]) * 25)
    lvl = int(rng.integers(1, 41))
    deaths = int(rng.poisson(max(0.5, 6 - lvl * 0.08) * rng.uniform(0.3, 2)))
    succ = float(np.clip(0.35 + lvl * 0.01 - deaths * 0.025 + rng.normal(0, 0.12), 0.05, 0.98))
    score = succ * 2.2 - deaths * 0.08 + lvl * 0.01 + rng.normal(0, 0.25)
    diff = "HARD" if score > 1.35 else ("EASY" if score < 0.85 else "MEDIUM")
    rows.append((lvl, deaths, round(succ, 3), *[round(float(x), 3) for x in r], diff))
pd.DataFrame(rows, columns=["player_level", "deaths", "success_rate", "combat_rate", "stealth_rate",
                            "exploration_rate", "dialogue_rate", "recommended_difficulty"]) \
    .to_csv(f"{OUT}/04_player_profile_dataset.csv", index=False)

# ---------------------------------------------------------------- 05 QUEST RECOMMENDATION
QUESTS = [("Slay the Goblin King", "COMBAT", "MEDIUM"), ("Arena of Blood", "COMBAT", "HARD"),
          ("Rat Cellar Cleanup", "COMBAT", "EASY"), ("Lost Temple Map", "EXPLORATION", "MEDIUM"),
          ("Whispering Woods", "EXPLORATION", "EASY"), ("Abyss Descent", "EXPLORATION", "HARD"),
          ("Peace Treaty", "DIALOGUE", "MEDIUM"), ("Settle the Feud", "DIALOGUE", "EASY"),
          ("Court Intrigue", "DIALOGUE", "HARD"), ("Midnight Heist", "STEALTH", "HARD"),
          ("Spy on the Baron", "STEALTH", "MEDIUM"), ("Shadow Courier", "STEALTH", "EASY"),
          ("Dragon's Hoard", "TREASURE", "HARD"), ("Buried Relic", "TREASURE", "MEDIUM")]
TYPE2PREF = {"COMBAT": "combat_pref", "EXPLORATION": "exploration_pref", "DIALOGUE": "dialogue_pref",
             "STEALTH": "stealth_pref", "TREASURE": "rare_weapon_pref"}
DIFFV = {"EASY": 0, "MEDIUM": 1, "HARD": 2}
rows = []
for pid in range(1, 301):
    prefs = {c: round(float(rng.beta(2, 2)), 2) for c in TYPE2PREF.values()}
    pref_diff = rng.uniform(0, 2)
    for qi in rng.choice(len(QUESTS), size=6, replace=False):
        qn, qt, qd = QUESTS[qi]
        p = 0.1 + 0.8 * prefs[TYPE2PREF[qt]] - 0.15 * abs(DIFFV[qd] - pref_diff)
        liked = int(rng.random() < np.clip(p + rng.normal(0, 0.05), 0.03, 0.97))
        rows.append((f"P{pid:03d}", *prefs.values(), qn, qt, qd, liked))
pd.DataFrame(rows, columns=["player_id", *TYPE2PREF.values(), "quest_name", "quest_type",
                            "quest_difficulty", "liked_quest"]) \
    .to_csv(f"{OUT}/05_quest_recommendation_dataset.csv", index=False)

print("Done. Files written to", os.path.abspath(OUT))
