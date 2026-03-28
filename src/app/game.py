"""
Phase 4: 2-player interactive Streamlit cricket match simulator.

Player 1 = batting team captain (picks openers, selects next batter after wickets)
Player 2 = bowling team captain (picks opening bowler, selects bowler each over)

Usage:
    streamlit run src/app/game.py
    streamlit run src/app/game.py -- --checkpoint checkpoints/qwen-cricket
"""

import sys
import argparse
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generate.commentary import get_commentary


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="gullyGPT",
    page_icon="🏏",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEAMS = {
    "India": ["Rohit Sharma", "Virat Kohli", "Shubman Gill", "Suryakumar Yadav",
              "Hardik Pandya", "Rishabh Pant", "Jasprit Bumrah", "Mohammed Shami",
              "Ravindra Jadeja", "Kuldeep Yadav", "Arshdeep Singh"],
    "Pakistan": ["Babar Azam", "Mohammad Rizwan", "Fakhar Zaman", "Iftikhar Ahmed",
                 "Shadab Khan", "Shaheen Afridi", "Naseem Shah", "Haris Rauf",
                 "Mohammad Nawaz", "Asif Ali", "Saim Ayub"],
    "Australia": ["David Warner", "Travis Head", "Steve Smith", "Glenn Maxwell",
                  "Mitchell Marsh", "Tim David", "Pat Cummins", "Mitchell Starc",
                  "Josh Hazlewood", "Adam Zampa", "Marcus Stoinis"],
    "England": ["Jos Buttler", "Jonny Bairstow", "Jason Roy", "Ben Stokes",
                "Liam Livingstone", "Moeen Ali", "Chris Woakes", "Jofra Archer",
                "Sam Curran", "Adil Rashid", "Mark Wood"],
    "West Indies": ["Nicholas Pooran", "Rovman Powell", "Brandon King", "Shimron Hetmyer",
                    "Andre Russell", "Kieron Pollard", "Sunil Narine", "Jason Holder",
                    "Alzarri Joseph", "Akeal Hosein", "Kyle Mayers"],
    "South Africa": ["Quinton de Kock", "Temba Bavuma", "Rassie van der Dussen", "David Miller",
                     "Heinrich Klaasen", "Kagiso Rabada", "Anrich Nortje", "Tabraiz Shamsi",
                     "Marco Jansen", "Aiden Markram", "Wayne Parnell"],
}

VENUES = ["Wankhede", "Eden Gardens", "MCG", "Lords", "Dubai International",
          "Gaddafi Stadium", "SCG", "Edgbaston", "Newlands", "Sabina Park"]

PITCHES = ["flat", "green", "dusty", "slow", "bouncy"]


def get_engine():
    if "engine" not in st.session_state:
        checkpoint = st.session_state.get("checkpoint", "checkpoints/qwen-cricket")
        from src.generate.engine import MatchEngine
        st.session_state.engine = MatchEngine(checkpoint)
    return st.session_state.engine


def init_state():
    defaults = {
        "phase": "setup",          # setup | innings1_pick_bowler | batting | innings_break | done
        "innings": 1,
        "over": 0,
        "legal_balls": 0,
        "total_balls": 0,
        "runs": [0, 0],
        "wickets": [0, 0],
        "balls_log": [],
        "batters": [[], []],       # [batting_order_inn1, batting_order_inn2]
        "batter_idx": [0, 0],
        "current_batter": ["", ""],
        "non_striker": ["", ""],
        "bowler": ["", ""],
        "target": None,
        "match_over": False,
        "result_str": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# UI Sections
# ---------------------------------------------------------------------------

def render_setup():
    st.title("🏏 gullyGPT")
    st.markdown("*Cricket as next-token prediction*")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Player 1 — Batting Team")
        team1 = st.selectbox("Select your team", list(TEAMS.keys()), key="team1")
        squad1 = TEAMS[team1]
        openers1 = st.multiselect(
            "Pick your 2 openers (striker first)",
            squad1, max_selections=2, key="openers1"
        )
        batting_order1 = st.multiselect(
            "Batting order (3-11)",
            [p for p in squad1 if p not in openers1],
            key="batting_order1"
        )

    with col2:
        st.subheader("Player 2 — Bowling Team")
        remaining_teams = [t for t in TEAMS if t != team1]
        team2 = st.selectbox("Select your team", remaining_teams, key="team2")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        venue = st.selectbox("Venue", VENUES, key="venue")
        pitch = st.selectbox("Pitch type", PITCHES, key="pitch")
    with col4:
        toss_winner = st.radio("Toss won by", [team1, team2], key="toss_winner")
        toss_decision = st.radio("Elected to", ["bat", "field"], key="toss_decision")

    st.divider()
    checkpoint = st.text_input("Model checkpoint path", value="checkpoints/qwen-cricket", key="checkpoint")

    ready = len(openers1) == 2
    if not ready:
        st.warning("Player 1: pick exactly 2 openers to start.")

    if st.button("🏏 Start Match", disabled=not ready, type="primary"):
        # Build full batting order
        full_order1 = openers1 + batting_order1 + [p for p in TEAMS[team1] if p not in openers1 + batting_order1]
        st.session_state.batters[0] = full_order1

        squad2 = TEAMS[team2]
        st.session_state.batters[1] = squad2  # Player 2 sets order implicitly via bowler picks

        # Init engine + match
        engine = get_engine()
        engine.start_match(
            team1, team2, venue, toss_winner, toss_decision, pitch=pitch
        )

        # Set first batting team based on toss
        if toss_decision == "bat":
            st.session_state.batting_first = team1
            st.session_state.bowling_first = team2
        else:
            st.session_state.batting_first = team2
            st.session_state.bowling_first = team1

        inn = 0  # index for batting_first
        st.session_state.current_batter[inn] = full_order1[0] if toss_decision == "bat" else squad2[0]
        st.session_state.non_striker[inn] = full_order1[1] if toss_decision == "bat" else squad2[1]
        st.session_state.batter_idx[inn] = 2

        st.session_state.phase = "pick_bowler"
        st.rerun()


def render_scoreboard():
    inn = st.session_state.innings - 1
    team_batting = st.session_state.batting_first if inn == 0 else st.session_state.bowling_first
    runs = st.session_state.runs[inn]
    wickets = st.session_state.wickets[inn]
    overs_done = st.session_state.over
    balls = st.session_state.legal_balls

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score", f"{runs}/{wickets}")
    col2.metric("Overs", f"{overs_done}.{balls}")
    col3.metric("Batting", team_batting)
    if st.session_state.target:
        needed = st.session_state.target - runs
        col4.metric("Target", f"{st.session_state.target} | Need {needed}")

    if st.session_state.balls_log:
        last = st.session_state.balls_log[-1]
        st.info(f"Last ball: {last}")
    if st.session_state.get("last_commentary"):
        st.markdown(f"> *{st.session_state['last_commentary']}*")


def render_pick_bowler():
    inn = st.session_state.innings - 1
    bowling_team_name = st.session_state.bowling_first if inn == 0 else st.session_state.batting_first
    squad = TEAMS[bowling_team_name]

    st.subheader(f"Player 2 — Pick bowler for Over {st.session_state.over + 1}")
    bowler = st.selectbox("Choose bowler", squad, key=f"bowler_pick_{st.session_state.over}")

    if st.button("Confirm bowler", type="primary"):
        st.session_state.bowler[inn] = bowler
        st.session_state.phase = "batting"
        st.rerun()


def render_batting():
    inn = st.session_state.innings - 1
    engine = get_engine()

    render_scoreboard()
    st.divider()

    batter = st.session_state.current_batter[inn]
    non_striker = st.session_state.non_striker[inn]
    bowler = st.session_state.bowler[inn]
    over = st.session_state.over
    ball_num = st.session_state.legal_balls

    st.markdown(f"**Over {over}.{ball_num + 1}** | {bowler} to {batter}")

    if st.button("⚡ Generate Next Ball", type="primary"):
        ball = engine.generate_ball(
            over=over,
            ball_idx=ball_num,
            bowler=bowler,
            batter=batter,
            non_striker=non_striker,
        )

        if ball is None:
            st.error("Model failed to generate a valid ball. Try again.")
            return

        # Commentary (async-friendly: fire and store before state mutates)
        from src.generate.engine import InningsState as _IS
        _state_snapshot = _IS(
            batting_team=st.session_state.batting_first if inn == 0 else st.session_state.bowling_first,
            bowling_team=st.session_state.bowling_first if inn == 0 else st.session_state.batting_first,
            runs=st.session_state.runs[inn],
            wickets=st.session_state.wickets[inn],
            legal_balls=st.session_state.legal_balls,
            overs_complete=st.session_state.over,
            target=st.session_state.target,
        )
        commentary = get_commentary(ball, _state_snapshot, innings_num=inn + 1)
        st.session_state["last_commentary"] = commentary

        # Update state
        st.session_state.runs[inn] += ball.runs
        st.session_state.total_balls += 1
        if ball.is_legal:
            st.session_state.legal_balls += 1
        st.session_state.balls_log.append(str(ball))

        # Wicket
        if ball.is_wicket:
            st.session_state.wickets[inn] += 1
            if st.session_state.wickets[inn] < 10:
                batting_squad = st.session_state.batters[inn]
                idx = st.session_state.batter_idx[inn]
                if idx < len(batting_squad):
                    st.session_state.current_batter[inn] = batting_squad[idx]
                    st.session_state.batter_idx[inn] += 1
                else:
                    st.session_state.current_batter[inn] = f"Batter {idx + 1}"

        # Rotate strike on odd legal runs
        if ball.runs % 2 == 1 and ball.is_legal:
            st.session_state.current_batter[inn], st.session_state.non_striker[inn] = (
                st.session_state.non_striker[inn], st.session_state.current_batter[inn]
            )

        # Check innings end conditions
        runs = st.session_state.runs[inn]
        wickets = st.session_state.wickets[inn]
        target = st.session_state.target

        inn_over = (target and runs >= target) or wickets >= 10

        if st.session_state.legal_balls >= 6:
            st.session_state.over += 1
            st.session_state.legal_balls = 0
            # Rotate strike at end of over
            st.session_state.current_batter[inn], st.session_state.non_striker[inn] = (
                st.session_state.non_striker[inn], st.session_state.current_batter[inn]
            )
            if st.session_state.over >= 20 or inn_over:
                inn_over = True
            else:
                st.session_state.phase = "pick_bowler"

        if inn_over:
            _end_innings(inn, engine)

        st.rerun()


def _end_innings(inn: int, engine):
    if inn == 0:
        # End of innings 1, set target
        st.session_state.target = st.session_state.runs[0] + 1
        st.session_state.innings = 2
        st.session_state.over = 0
        st.session_state.legal_balls = 0
        st.session_state.phase = "innings_break"
        # Pin inn1 result into model context so it knows the target in inn2
        engine.set_innings1_result(
            st.session_state.runs[0],
            st.session_state.wickets[0],
            20,
        )

        # Set up innings 2 openers
        batting2_squad = st.session_state.batters[1]
        st.session_state.current_batter[1] = batting2_squad[0]
        st.session_state.non_striker[1] = batting2_squad[1]
        st.session_state.batter_idx[1] = 2

        engine.start_innings(2, st.session_state.bowling_first)
    else:
        # Match over
        runs1 = st.session_state.runs[0]
        runs2 = st.session_state.runs[1]
        wickets2 = st.session_state.wickets[1]

        if runs2 >= st.session_state.target:
            winner = st.session_state.bowling_first
            by = f"{10 - wickets2} wickets"
            by_tag = f"{10 - wickets2}wickets"
        else:
            winner = st.session_state.batting_first
            by = f"{runs1 - runs2} runs"
            by_tag = f"{runs1 - runs2}runs"

        engine.add_result(winner, by_tag)
        st.session_state.result_str = f"{winner} won by {by}"
        st.session_state.phase = "done"


def render_innings_break():
    st.title("🔄 Innings Break")
    inn1_team = st.session_state.batting_first
    st.metric(f"{inn1_team} scored", f"{st.session_state.runs[0]}/{st.session_state.wickets[0]}")
    st.metric("Target", st.session_state.target)
    st.markdown(f"**{st.session_state.bowling_first}** need **{st.session_state.target}** to win.")

    if st.button("Start Innings 2 →", type="primary"):
        st.session_state.phase = "pick_bowler"
        st.rerun()


def render_done():
    st.balloons()
    st.title("🏆 Match Over")
    st.success(st.session_state.result_str)
    st.divider()
    st.subheader("Ball-by-ball log")
    for b in st.session_state.balls_log:
        st.text(b)

    if st.button("Play Again"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_state()
    phase = st.session_state.get("phase", "setup")

    if phase == "setup":
        render_setup()
    elif phase == "pick_bowler":
        render_pick_bowler()
    elif phase == "batting":
        render_batting()
    elif phase == "innings_break":
        render_innings_break()
    elif phase == "done":
        render_done()


if __name__ == "__main__":
    main()
