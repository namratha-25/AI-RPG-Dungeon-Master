"""AI RPG Dungeon Master - game engine.

Closed loop (matches the deck):
  player text -> intent ML + sentiment ML -> NPC behaviour ML -> world-state update
  -> player profiling (K-Means) + difficulty ML + quest recommender
  -> memory (short-term + long-term retrieval) -> narration

Narration is template-based here: the narrator ONLY describes the result. It never changes state.
(An LLM can be dropped in later by replacing DungeonMaster.narrate().)
"""
import os
import json
import random
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = os.path.join(ROOT, "models")
SAVE_PATH = os.path.join(ROOT, "savegame.json")

LOCATIONS = ["tavern", "dungeon", "castle", "forest", "village", "mountain", "market"]
CATEGORY = {"attack": "combat", "intimidate": "combat", "steal": "stealth", "flee": "stealth",
            "explore": "exploration", "negotiate": "dialogue", "persuade": "dialogue", "help": "dialogue"}
ACTION_VERB = {"attack": "strike at", "negotiate": "bargain with", "steal": "try to rob", "explore": "search near",
               "flee": "run from", "help": "come to the aid of", "persuade": "try to win over",
               "intimidate": "menace"}

DEFAULT_NPCS = {
    "Captain Aldric": dict(type="guard",    trust=0.50, fear=0.20, anger=0.20, loyalty=0.70, greed=0.20),
    "Mira the Merchant": dict(type="merchant", trust=0.55, fear=0.30, anger=0.10, loyalty=0.40, greed=0.80),
    "Father Joren": dict(type="priest",   trust=0.70, fear=0.20, anger=0.05, loyalty=0.80, greed=0.05),
    "Kest the Brigand": dict(type="bandit",   trust=0.20, fear=0.30, anger=0.60, loyalty=0.30, greed=0.70),
    "Lord Valen": dict(type="noble",    trust=0.40, fear=0.30, anger=0.30, loyalty=0.50, greed=0.60),
    "Old Tam": dict(type="villager", trust=0.60, fear=0.50, anger=0.10, loyalty=0.50, greed=0.20),
}

REACTION_LINES = {
    "help": ["{npc} nods and lends a hand.", "{npc} smiles warmly and offers what help they can.",
             "\"Well met, traveler,\" says {npc}, stepping aside to assist you."],
    "ignore": ["{npc} barely glances at you and turns away.", "{npc} pays you no mind.",
               "{npc} shrugs, unmoved by what you did."],
    "threaten": ["{npc} snarls a warning: \"Do that again and you'll regret it.\"",
                 "{npc}'s hand drifts to a weapon. \"Watch yourself.\"",
                 "{npc} glares at you, jaw clenched."],
    "negotiate": ["{npc} narrows their eyes and names a price.", "{npc} leans in. \"Let's talk terms.\"",
                  "\"Everything has a cost,\" {npc} says with a thin smile."],
    "attack": ["{npc} lashes out at you!", "{npc} draws steel and attacks!",
               "{npc} answers with violence and strikes back!"],
    "flee": ["{npc} panics and scrambles away.", "{npc} bolts, terror in their eyes.",
             "{npc} backs off and flees from you."],
}
OPENERS = {
    "attack": "You {verb} {npc}.", "negotiate": "You {verb} {npc}.", "steal": "You {verb} {npc}.",
    "explore": "You {verb} {npc}, studying your surroundings in the {loc}.",
    "flee": "You {verb} {npc}.", "help": "You {verb} {npc}.", "persuade": "You {verb} {npc}.",
    "intimidate": "You {verb} {npc}.",
}


class DungeonMaster:
    def __init__(self, models_dir=MODELS, seed=7):
        self.rng = random.Random(seed)
        m = lambda n: joblib.load(os.path.join(models_dir, n))
        self.intent_m, self.sent_m = m("intent_model.joblib"), m("sentiment_model.joblib")
        self.npc_m, self.diff_m = m("npc_behavior_model.joblib"), m("difficulty_model.joblib")
        self.arch_m, self.quest_m = m("archetype_kmeans.joblib"), m("quest_model.joblib")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.world = dict(reputation=0, town_alert=0, dragon_stir=0, dragon_awake=False,
                          guards_hunting=False, gold=50, turn=0, location="tavern")
        self.npcs = {n: dict(d, hp=100, alive=True) for n, d in DEFAULT_NPCS.items()}
        self.player = dict(level=1, deaths=0, n_actions=0, n_success=0,
                           counts=dict(combat=0, stealth=0, exploration=0, dialogue=0))
        self.active_quests, self.done_quests = [], []
        self.short_term = deque(maxlen=5)
        self.long_term = []          # key events that persist across restarts
        self.last_events = []

    def save(self, path=SAVE_PATH):
        data = dict(world=self.world, npcs=self.npcs, player=self.player, active_quests=self.active_quests,
                    done_quests=self.done_quests, short_term=list(self.short_term), long_term=self.long_term)
        with open(path, "w") as f:
            json.dump(data, f, indent=1)

    def load(self, path=SAVE_PATH):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            d = json.load(f)
        self.world, self.npcs, self.player = d["world"], d["npcs"], d["player"]
        self.active_quests, self.done_quests = d["active_quests"], d["done_quests"]
        self.short_term = deque(d["short_term"], maxlen=5)
        self.long_term = d["long_term"]
        return True

    # ------------------------------------------------------------------ ML helpers
    @staticmethod
    def _predict_text(bundle, text):
        proba = bundle["model"].predict_proba([text])[0]
        order = np.argsort(proba)[::-1]
        classes = bundle["le"].classes_
        return [(classes[i], float(proba[i])) for i in order]

    def predict_intent(self, text):
        return self._predict_text(self.intent_m, text)

    def predict_sentiment(self, text):
        return self._predict_text(self.sent_m, text)

    def predict_npc_reaction(self, npc_name, intent, sentiment):
        n = self.npcs[npc_name]
        row = dict(npc_type=n["type"], player_intent=intent, player_sentiment=sentiment,
                   trust=n["trust"], fear=n["fear"], anger=n["anger"], loyalty=n["loyalty"], greed=n["greed"])
        b = self.npc_m
        X = pd.DataFrame([row])[b["cat"] + b["num"]]
        proba = b["model"].predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        return [(b["le"].classes_[i], float(proba[i])) for i in order]

    def relationship(self, npc_name, sentiment="neutral"):
        n = self.npcs[npc_name]
        sv = {"positive": 0.4, "neutral": 0.0, "negative": -0.4}[sentiment]
        return float(np.clip((n["trust"] + n["loyalty"] - n["anger"] - n["fear"] * 0.3) * 60 + sv * 40, -100, 100))

    # ------------------------------------------------------------------ player profile
    def rates(self):
        c = self.player["counts"]
        tot = sum(c.values()) + 2.0                       # smoothing -> starts near uniform
        return {k: (v + 0.5) / tot for k, v in c.items()}

    def archetype(self):
        if self.player["n_actions"] < 3:
            return "Unknown (need 3+ actions)"
        r = self.rates()
        X = np.array([[r["combat"], r["stealth"], r["exploration"], r["dialogue"]]])
        return self.arch_m["names"][int(self.arch_m["kmeans"].predict(X)[0])]

    def success_rate(self):
        return (self.player["n_success"] + 1.5) / (self.player["n_actions"] + 3.0)

    def recommended_difficulty(self):
        b = self.diff_m
        X = pd.DataFrame([dict(player_level=self.player["level"], deaths=self.player["deaths"],
                               success_rate=self.success_rate())])[b["num"]]
        return str(b["le"].inverse_transform(b["model"].predict(X))[0])

    def record_death(self):
        self.player["deaths"] += 1

    # ------------------------------------------------------------------ quests
    def recommend_quests(self, top_n=3):
        r, qm = self.rates(), self.quest_m
        prefs = dict(combat_pref=min(1, r["combat"] * 2), exploration_pref=min(1, r["exploration"] * 2),
                     dialogue_pref=min(1, r["dialogue"] * 2), stealth_pref=min(1, r["stealth"] * 2),
                     rare_weapon_pref=min(1, r["exploration"] + r["stealth"]))
        type2pref = {"COMBAT": "combat_pref", "EXPLORATION": "exploration_pref", "DIALOGUE": "dialogue_pref",
                     "STEALTH": "stealth_pref", "TREASURE": "rare_weapon_pref"}
        cand = qm["quests"].copy()
        taken = set(self.active_quests) | set(self.done_quests)
        cand = cand[~cand["quest_name"].isin(taken)].copy()
        if cand.empty:
            return []
        for c, v in prefs.items():
            cand[c] = v
        cand["matched_pref"] = [prefs[type2pref[t]] for t in cand["quest_type"]]
        cand["score"] = qm["model"].predict_proba(cand[qm["features"]])[:, 1]
        want = self.recommended_difficulty()
        cand["fits_difficulty"] = cand["quest_difficulty"] == want
        cand["final"] = cand["score"] * np.where(cand["fits_difficulty"], 1.0, 0.9)
        top = cand.sort_values("final", ascending=False).head(top_n)
        return [dict(name=r_.quest_name, type=r_.quest_type, difficulty=r_.quest_difficulty,
                     score=float(r_.score), fits_difficulty=bool(r_.fits_difficulty)) for r_ in top.itertuples()]

    def accept_quest(self, name):
        if name not in self.active_quests and name not in self.done_quests:
            self.active_quests.append(name)

    def complete_quest(self, name):
        if name in self.active_quests:
            self.active_quests.remove(name)
            self.done_quests.append(name)
            self.world["gold"] += 50
            self.world["reputation"] = int(np.clip(self.world["reputation"] + 3, -100, 100))

    # ------------------------------------------------------------------ memory
    def _remember(self, text, npc, turn):
        self.long_term.append(dict(turn=turn, npc=npc, text=text))

    def _recall(self, npc, query):
        """Semantic-ish retrieval over long-term memory (TF-IDF cosine + same-NPC bonus)."""
        past = [m for m in self.long_term if m["turn"] < self.world["turn"]]
        if not past:
            return None
        docs = [m["text"] for m in past]
        vec = TfidfVectorizer().fit(docs + [query])
        sims = cosine_similarity(vec.transform([query]), vec.transform(docs))[0]
        scores = [s + (0.5 if m["npc"] == npc else 0) for s, m in zip(sims, past)]
        i = int(np.argmax(scores))
        return past[i] if scores[i] >= 0.5 else None

    # ------------------------------------------------------------------ the turn
    def take_turn(self, text, npc_name, location=None, min_conf=0.25):
        text = text.strip()
        if not text:
            return dict(ok=False, narration="The Dungeon Master waits for you to act.")
        if not self.npcs[npc_name]["alive"]:
            return dict(ok=False, narration=f"{npc_name} is already dead. Choose someone else.")
        if location:
            self.world["location"] = location
        loc = self.world["location"]

        intents, sents = self.predict_intent(text), self.predict_sentiment(text)
        intent, iconf = intents[0]
        if iconf < min_conf:
            return dict(ok=False, intents=intents[:3],
                        narration="The Dungeon Master tilts their head - your intent is unclear. "
                                  "Try an action like 'I attack the guard' or 'I offer a trade'.")
        sentiment, sconf = sents[0]

        self.world["turn"] += 1
        turn = self.world["turn"]
        npc = self.npcs[npc_name]
        events = []

        # 1) NPC behaviour (ML) -> reaction
        reactions = self.predict_npc_reaction(npc_name, intent, sentiment)
        reaction = reactions[0][0]

        # 2) update NPC traits from the player's action (rule layer)
        d = dict(trust=0.0, fear=0.0, anger=0.0, loyalty=0.0)
        if intent in ("help", "persuade"):
            d.update(trust=0.10, anger=-0.05, loyalty=0.05)
        elif intent == "attack":
            d.update(trust=-0.15, anger=0.20, fear=0.10, loyalty=-0.05)
        elif intent == "steal":
            d.update(trust=-0.20, anger=0.10)
        elif intent == "intimidate":
            d.update(trust=-0.10, anger=0.10, fear=0.20)
        elif intent == "negotiate":
            d.update(trust=0.03)
        if sentiment == "positive":
            d["trust"] += 0.03
        elif sentiment == "negative":
            d["trust"] -= 0.03
            d["anger"] += 0.03
        for k, v in d.items():
            npc[k] = float(np.clip(npc[k] + v, 0, 1))
        rel = self.relationship(npc_name, sentiment)

        # 3) world-state update
        w, hostile = self.world, intent in ("attack", "steal", "intimidate")
        if intent == "attack":
            npc["hp"] -= 10 + 2 * self.player["level"]
            w["reputation"] += 3 if npc["type"] == "bandit" else -8
            w["town_alert"] += 0 if npc["type"] == "bandit" else 12
            if npc["hp"] <= 0:
                npc["alive"] = False
                events.append(f"{npc_name} has been slain.")
                w["town_alert"] += 0 if npc["type"] == "bandit" else 20
                w["reputation"] += 5 if npc["type"] == "bandit" else -15
                self._remember(f"you killed {npc_name} the {npc['type']} in the {loc}", npc_name, turn)
        elif intent == "steal":
            w["reputation"] -= 6
            w["town_alert"] += 10 + (10 if reaction in ("attack", "threaten") else 0)
        elif intent == "intimidate":
            w["reputation"] -= 3
            w["town_alert"] += 5
        elif intent in ("help", "persuade"):
            w["reputation"] += 5
            w["town_alert"] -= 3
        elif intent == "negotiate" and reaction in ("negotiate", "help"):
            w["gold"] += 10 if npc["type"] == "merchant" else 3
        if loc in ("mountain", "dungeon") and intent in ("attack", "explore"):
            w["dragon_stir"] += 15
            if w["dragon_stir"] >= 60 and not w["dragon_awake"]:
                w["dragon_awake"] = True
                events.append("A roar shakes the mountain - the DRAGON HAS AWAKENED!")
                self._remember("the dragon awoke because of your noise", "-", turn)
        for k, lo, hi in (("reputation", -100, 100), ("town_alert", 0, 100), ("dragon_stir", 0, 100)):
            w[k] = int(np.clip(w[k], lo, hi))
        if w["town_alert"] >= 70 and not w["guards_hunting"]:
            w["guards_hunting"] = True
            events.append("Alarm bells ring - the town guards are now hunting you!")
        elif w["town_alert"] < 40:
            w["guards_hunting"] = False

        # 4) player profile (updates behaviour counts, success, level)
        p = self.player
        p["counts"][CATEGORY[intent]] += 1
        p["n_actions"] += 1
        p["n_success"] += int(reaction in ("help", "negotiate", "ignore", "flee"))
        if p["n_actions"] % 5 == 0:
            p["level"] += 1
            events.append(f"You feel stronger - level {p['level']}!")

        # 5) memory: key events go to long-term; everything goes to short-term
        recalled = self._recall(npc_name, f"{npc_name} {npc['type']} {intent} {text}")
        if intent in ("attack", "steal", "help", "intimidate", "persuade") and npc["alive"]:
            past_tense = {"attack": "attacked", "steal": "tried to rob", "help": "helped",
                          "intimidate": "threatened", "persuade": "won over"}[intent]
            self._remember(f"you {past_tense} {npc_name} the {npc['type']} in the {loc}", npc_name, turn)
        self.short_term.append(dict(turn=turn, text=text, npc=npc_name, intent=intent, reaction=reaction))

        res = dict(ok=True, turn=turn, intent=intent, intent_conf=iconf, intents=intents[:3],
                   sentiment=sentiment, sentiment_conf=sconf, reactions=reactions[:3], reaction=reaction,
                   relationship=rel, events=events, recalled=recalled["text"] if recalled else None,
                   archetype=self.archetype(), difficulty=self.recommended_difficulty())
        res["narration"] = self.narrate(res, npc_name, loc)
        self.last_events = events
        return res

    # ------------------------------------------------------------------ narration (read-only)
    def narrate(self, res, npc_name, loc):
        r = random.Random(res["turn"])
        parts = [OPENERS[res["intent"]].format(verb=ACTION_VERB[res["intent"]], npc=npc_name, loc=loc)]
        parts.append(r.choice(REACTION_LINES[res["reaction"]]).format(npc=npc_name))
        if res["recalled"]:
            parts.append(f"{npc_name} has not forgotten: {res['recalled']}.")
        rel = res["relationship"]
        mood = ("regards you as a trusted ally" if rel > 40 else "seems wary of you" if rel > -10
                else "clearly despises you")
        parts.append(f"({npc_name} {mood}.)")
        parts.extend(res["events"])
        return " ".join(parts)

    def world_snapshot(self):
        return dict(self.world, level=self.player["level"], deaths=self.player["deaths"],
                    success_rate=round(self.success_rate(), 2), archetype=self.archetype(),
                    difficulty=self.recommended_difficulty())

    def npc_table(self):
        rows = []
        for n, d in self.npcs.items():
            rows.append(dict(NPC=n, type=d["type"], alive=d["alive"], hp=max(0, d["hp"]),
                             relationship=round(self.relationship(n), 1), trust=round(d["trust"], 2),
                             fear=round(d["fear"], 2), anger=round(d["anger"], 2)))
        return pd.DataFrame(rows)
