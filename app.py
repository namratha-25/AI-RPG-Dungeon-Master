"""AI RPG Dungeon Master - Streamlit UI.   Run:  streamlit run app.py"""
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from engine import DungeonMaster, LOCATIONS  # noqa: E402

st.set_page_config(page_title="AI RPG Dungeon Master", page_icon="🐉", layout="wide")


if "dm" not in st.session_state:
    st.session_state.dm = DungeonMaster()
    st.session_state.log = []
dm = st.session_state.dm

st.title("🐉 AI RPG Dungeon Master")
st.caption("ML models decide what happens; the narrator only describes it.")

# ----------------------------------------------------------------- sidebar: world + player
with st.sidebar:
    st.header("World state")
    w = dm.world
    c1, c2 = st.columns(2)
    c1.metric("Turn", w["turn"])
    c2.metric("Gold", w["gold"])
    c1.metric("Reputation", w["reputation"])
    c2.metric("Town alert", w["town_alert"])
    st.progress(w["dragon_stir"] / 100, text=f"Dragon stirring: {w['dragon_stir']}/100")
    st.write(f"🐲 Dragon awake: **{w['dragon_awake']}**  |  🚨 Guards hunting: **{w['guards_hunting']}**")

    st.header("Player")
    snap = dm.world_snapshot()
    st.write(f"Level **{snap['level']}**  |  Deaths **{snap['deaths']}**  |  Success rate **{snap['success_rate']:.0%}**")
    st.write(f"Archetype (K-Means): **{snap['archetype']}**")
    st.write(f"Recommended difficulty: **{snap['difficulty']}**")
    r = dm.rates()
    st.bar_chart(pd.Series({k.title(): v for k, v in r.items()}), height=140)
    if st.button("☠️ I died (+1 death)"):
        dm.record_death()
        st.rerun()

    st.header("Game")
    b1, b2, b3 = st.columns(3)
    if b1.button("Save"):
        dm.save()
        st.toast("Game saved")
    if b2.button("Load"):
        ok = dm.load()
        st.session_state.log = []
        st.toast("Loaded save" if ok else "No save file yet")
        st.rerun()
    if b3.button("Reset"):
        dm.reset()
        st.session_state.log = []
        st.rerun()

tab_play, tab_models = st.tabs(["🎲 Play", "📊 Models & evaluation"])

# ----------------------------------------------------------------- play tab
with tab_play:
    left, right = st.columns([3, 2])
    with left:
        alive = [n for n, d in dm.npcs.items() if d["alive"]]
        with st.form("turn", clear_on_submit=True):
            f1, f2 = st.columns(2)
            npc = f1.selectbox("Who are you acting on?", alive or list(dm.npcs))
            loc = f2.selectbox("Location", LOCATIONS, index=LOCATIONS.index(dm.world["location"]))
            text = st.text_input("What do you do?", placeholder="e.g. I attack the guard / Please help me, friend")
            go = st.form_submit_button("Act ⚔️")
        if go and text:
            res = dm.take_turn(text, npc, loc)
            st.session_state.log.append((text, npc, res))
            st.rerun()

        if not st.session_state.log:
            st.info("Pick an NPC, type an action in plain English, and watch the world react.")
        for text_, npc_, res in reversed(st.session_state.log):
            with st.container(border=True):
                st.markdown(f"**You → {npc_}:** {text_}")
                st.markdown(res["narration"])
                if res.get("ok"):
                    with st.expander("🔍 What the ML models decided"):
                        st.write(f"**Intent:** {res['intent']} ({res['intent_conf']:.0%})  - top 3: "
                                 + ", ".join(f"{a} {b:.0%}" for a, b in res["intents"]))
                        st.write(f"**Sentiment:** {res['sentiment']} ({res['sentiment_conf']:.0%})")
                        st.write("**NPC reaction:** " + ", ".join(f"{a} {b:.0%}" for a, b in res["reactions"]))
                        st.write(f"**Relationship score:** {res['relationship']:.0f}")
                        st.write(f"**Memory recalled:** {res['recalled'] or '-'}")

    with right:
        st.subheader("NPCs")
        st.dataframe(dm.npc_table(), hide_index=True)

        st.subheader("Quest recommendations")
        for i, q in enumerate(dm.recommend_quests(3)):
            with st.container(border=True):
                fit = "✅ fits your difficulty" if q["fits_difficulty"] else ""
                st.markdown(f"**{q['name']}** · {q['type']} · {q['difficulty']}  \nMatch score {q['score']:.0%} {fit}")
                if st.button("Accept", key=f"acc_{q['name']}"):
                    dm.accept_quest(q["name"])
                    st.rerun()
        if dm.active_quests:
            st.markdown("**Active quests**")
            for qn in list(dm.active_quests):
                if st.button(f"Complete: {qn}", key=f"done_{qn}"):
                    dm.complete_quest(qn)
                    st.rerun()
        if dm.done_quests:
            st.caption("Completed: " + ", ".join(dm.done_quests))

        st.subheader("Memory")
        st.caption("Short-term (last 5 turns)")
        for m in reversed(list(dm.short_term)):
            st.write(f"T{m['turn']}: {m['intent']} → {m['npc']} ({m['reaction']})")
        st.caption("Long-term (persists across restarts)")
        for m in reversed(dm.long_term[-6:]):
            st.write(f"T{m['turn']}: {m['text']}")

# ----------------------------------------------------------------- models tab
with tab_models:
    RES = os.path.join(ROOT, "results")
    st.write("Every candidate algorithm is trained and compared. Only the best model per task is deployed.")
    tasks = [("Player intent", "intent"), ("Sentiment", "sentiment"), ("NPC behaviour", "npc_behavior"),
             ("Difficulty", "difficulty")]
    for title, key in tasks:
        csv = os.path.join(RES, f"{key}_results.csv")
        if os.path.exists(csv):
            st.subheader(title)
            c1, c2 = st.columns([3, 2])
            df = pd.read_csv(csv)
            c1.dataframe(df, hide_index=True)
            c1.success(f"Deployed: {df.iloc[0]['model']}")
            png = os.path.join(RES, f"{key}_confusion.png")
            if os.path.exists(png):
                c2.image(png)
    st.subheader("Player archetypes (clustering)")
    c1, c2 = st.columns([3, 2])
    for f in ("kmeans_k_selection.csv", "clustering_comparison.csv"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            c1.dataframe(pd.read_csv(p), hide_index=True)
    p = os.path.join(RES, "archetype_clusters.png")
    if os.path.exists(p):
        c2.image(p)
    p = os.path.join(RES, "quest_results.csv")
    if os.path.exists(p):
        st.subheader("Quest recommendation")
        st.dataframe(pd.read_csv(p), hide_index=True)
