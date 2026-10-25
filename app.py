import secrets
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

APP_NAME = "Loop"  # placeholder name: change here and it updates everywhere
st.set_page_config(page_title=f"{APP_NAME} | a kinder, greener workplace", page_icon="🌿", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(135deg,#E3F4EA,#F6FBF8);padding:18px 22px;border-radius:16px;margin-bottom:12px}
.hero h2{margin:0;color:#1F6B49}.hero p{margin:4px 0 0;color:#476255}
div[data-testid="stMetric"]{background:#F1F7F3;border-radius:14px;padding:10px 14px}
.chip{display:inline-block;background:#E3F4EA;color:#1F6B49;border-radius:12px;padding:2px 10px;font-size:12px;margin-right:6px}
</style>""", unsafe_allow_html=True)

# ---------------- constants ----------------
DEPTS = ["Engineering", "Sales", "Operations", "Design", "Support", "Finance"]
ZONES = ["North", "South", "East", "West", "Central"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
EF = {"Drive alone": 0.17, "Carpool": 0.17, "Two-wheeler": 0.06, "Bus": 0.09,
      "Metro/train": 0.04, "Walk/cycle": 0.0, "Work from home": 0.0}  # kg CO2e per km (approximate)
SOURCE = {"Drive alone": "Parking-gate data", "Carpool": "Pool confirmation", "Two-wheeler": "Parking-gate data",
          "Bus": "Shuttle swipe", "Metro/train": "Transit-pass tap", "Walk/cycle": "Campus entry (no vehicle)",
          "Work from home": "HRMS attendance"}
FEATS = {"overtime": "Overtime hours", "meetings": "Meeting load", "pulse": "Pulse score",
         "km": "Commute distance", "wfh": "WFH days", "leave": "Leave taken"}
FIX = {"overtime": "Rebalance workload / cap overtime", "meetings": "Add meeting-free blocks",
       "pulse": "Run a listening session", "km": "Offer shuttle, carpool or hybrid days",
       "wfh": "Offer more hybrid flexibility", "leave": "Encourage managers to nudge leave"}
THEMES = {"Pool Party Week": (["Carpool"], 10, "Share a ride, share the credit."),
          "Transit Week": (["Bus", "Metro/train"], 20, "Let someone else do the driving."),
          "Walk & Wheel Week": (["Walk/cycle"], 4, "Legs, bikes, good vibes."),
          "Skip-the-Drive Week": (["Carpool", "Two-wheeler", "Bus", "Metro/train", "Walk/cycle", "Work from home"], 48,
                                  "Anything but driving solo.")}
TAGS = {"Workload": ["overtime", "workload", "hours", "deadline", "comp-off"],
        "Flexibility": ["wfh", "remote", "hybrid", "flex"],
        "Green & commute": ["carpool", "shuttle", "green", "parking", "transit", "commute"],
        "Growth": ["promotion", "growth", "learning", "training", "budget"]}


# ---------------- synthetic data (stands in for HRMS / gate / shuttle feeds) ----------------
@st.cache_data
def make_data(n=240, seed=7):
    rng = np.random.default_rng(seed)
    first = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vihaan", "Anaya",
             "Arjun", "Nisha", "Kiran", "Sana", "Dev", "Tara", "Neil", "Riya"]
    last = list("ABCDEFGHJKLMNPR")
    df = pd.DataFrame({
        "id": range(n), "name": [f"{first[i % 16]} {last[(i // 16) % 15]}." for i in range(n)],
        "dept": rng.choice(DEPTS, n), "gender": rng.choice(["Female", "Male", "Non-binary"], n, p=[.42, .55, .03]),
        "age_band": rng.choice(["<30", "30-40", "40+"], n, p=[.4, .4, .2]), "zone": rng.choice(ZONES, n),
        "shift": rng.choice([9, 9.5, 10], n, p=[.4, .3, .3]), "km": rng.integers(3, 32, n),
        "overtime": np.clip(rng.normal(6, 3, n), 0, 20).round(1), "meetings": np.clip(rng.normal(14, 5, n), 2, 30).round(1),
        "pulse": np.clip(rng.normal(3.4, .9, n), 1, 5).round(1), "wfh": rng.integers(0, 3, n), "leave": rng.integers(0, 6, n)})
    df["team"] = df.dept + "-" + rng.integers(1, 4, n).astype(str)
    z = (-1 + .18 * (df.overtime - 6) + .10 * (df.meetings - 14) - .8 * (df.pulse - 3.4) + .03 * (df.km - 17)
         - .3 * (df.wfh - 1) - .15 * (df.leave - 2.5) + rng.normal(0, .5, n))
    df["burnout"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    modes, p, rows = list(EF), [.38, .12, .14, .12, .10, .04, .10], []
    for r in df.itertuples():
        for d in DAYS:
            m = rng.choice(modes, p=p)
            occ = int(rng.integers(2, 5)) if m == "Carpool" else 1
            dist = 0 if m == "Work from home" else 2 * r.km
            solo, em = .17 * 2 * r.km, EF[m] * dist / occ
            rows.append((r.id, r.team, r.dept, d, m, occ, em, max(solo - em, 0), SOURCE[m], m != "Drive alone"))
    trips = pd.DataFrame(rows, columns=["id", "team", "dept", "day", "mode", "occ", "emitted", "saved", "source", "green"])
    return df, trips


@st.cache_data
def run_model(df):
    Xs = StandardScaler().fit_transform(df[list(FEATS)])
    lr = LogisticRegression().fit(Xs, df.burnout)
    return lr.predict_proba(Xs)[:, 1], lr.coef_[0], Xs


df, trips = make_data()
prob, coef, Xs = run_model(df)
df = df.assign(risk=prob)

S = st.session_state
S.setdefault("pools", {}); S.setdefault("pulse", {}); S.setdefault("voted", set()); S.setdefault("bonus", {})
S.setdefault("theme", "Pool Party Week"); S.setdefault("nudge_cap", 2); S.setdefault("thr", 0.6)
S.setdefault("questions", [
    {"q": "Can we get a shuttle from the East zone?", "votes": 18},
    {"q": "Can deadline weeks come with comp-off days?", "votes": 27},
    {"q": "More WFH flexibility for people with long commutes?", "votes": 21},
    {"q": "Can there be a learning budget per person?", "votes": 14}])
S.setdefault("actions", [
    {"said": "Too many back-to-back meetings", "did": "30-minute buffer is now the default", "status": "Done"},
    {"said": "Reserved parking for carpoolers", "did": "Spots marked at Gate 2", "status": "Done"},
    {"said": "Shuttle from the East zone", "did": "Pilot route under review", "status": "In progress"}])


S.setdefault("concerns", [
    {"code": "K7M2QA", "category": "Workload", "text": "Deadline weeks keep spilling into weekends and nobody tracks it.", "status": "In review", "reply": "", "date": "20 Oct"},
    {"code": "P4X9TD", "category": "Commute or facilities", "text": "The evening shuttle from the East zone leaves too early for people on the late shift.", "status": "New", "reply": "", "date": "22 Oct"}])


# ---------------- shared calculations ----------------
def fairness(thr):
    d = df.assign(flag=df.risk >= thr)
    out = {}
    for col in ["gender", "age_band"]:
        out[col] = d.groupby(col).flag.mean().mul(100).round(1)
    gap = max(s.max() - s.min() for s in out.values())
    return out, gap


def team_progress(theme):
    modes, goal, _ = THEMES[theme]
    t = trips[trips["mode"].isin(modes)].groupby("team").size().reindex(sorted(df.team.unique()), fill_value=0)
    if theme == "Pool Party Week":
        for eid, mates in S.pools.items():
            t[df.loc[eid, "team"]] += 3 * (1 if mates else 0)
    return t, goal


def wsi():
    well = (1 - df.risk.mean()) * 100
    green = trips.green.mean() * 100
    _, gap = fairness(S.thr)
    fair = max(0, 100 - 2 * gap)
    voice = 58 + min(len(S.voted) + 2 * len(S.pulse), 20)
    return {"Wellbeing": well, "Green participation": green, "Fairness": fair, "Voice (Townhall)": voice}


def tag(q):
    ql = q.lower()
    return [k for k, ws in TAGS.items() if any(w in ql for w in ws)] or ["General"]


def tree_days(kg):
    return kg / 0.0575  # one tree absorbs roughly 21 kg CO2 a year


# ---------------- employee view ----------------
def townhall_employee(e):
    st.subheader("Townhall corner 🎤")
    st.markdown("**Next townhall:** Thu 5 Nov 2026, 4:00 PM  ·  the agenda is built from *your* top questions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Ask anything (anonymous)")
        q = st.text_input("Your question", key="newq", placeholder="e.g. Can we get a shuttle from my area?")
        if st.button("Post question") and q.strip():
            S.questions.append({"q": q.strip(), "votes": 1}); S.voted.add(f"{e.id}-new-{len(S.questions)}")
            st.success("Posted. Nobody sees who asked.")
        st.markdown("##### Upvote what matters")
        for i, item in enumerate(sorted(S.questions, key=lambda x: -x["votes"])):
            a, b = st.columns([5, 1])
            a.write(f"{item['q']}  \n<span class='chip'>{item['votes']} votes</span>", unsafe_allow_html=True)
            if b.button("👍", key=f"v{i}") and f"{e.id}-{item['q']}" not in S.voted:
                item["votes"] += 1; S.voted.add(f"{e.id}-{item['q']}"); S.bonus[e.id] = S.bonus.get(e.id, 0) + 5
                st.rerun()
    with c2:
        st.markdown("##### You said, we did")
        for a in S.actions:
            st.write(f"**{a['said']}** → {a['did']}  <span class='chip'>{a['status']}</span>", unsafe_allow_html=True)


PULSE = {"😵 Drowning": 1.5, "😬 Heavy": 2.5, "🙂 Fine": 3.8, "😎 Light": 4.8}
TIPS = {"Overtime": "Pick one evening this week to log off on time, and tell your team in advance.",
        "Meeting load": "Ask for a no-meeting block, or turn one status meeting into a short written update.",
        "Long commute": "Check the Carpool tab or ask for a hybrid day: less time on the road means more rest.",
        "Little time off": "You have unused leave. Even one long weekend helps, so plan it before the quarter gets busy.",
        "How you feel": "You said things feel heavy. Talking helps, and you can do it anonymously in Speak up."}
CATS = ["Workload", "Manager or team", "Harassment or safety", "Burnout or wellbeing", "Commute or facilities", "Something else"]


def strain(e):
    pulse = PULSE.get(S.pulse.get(e.id), e.pulse)
    parts = {"Overtime": 30 * min(e.overtime / 15, 1), "Meeting load": 25 * min(e.meetings / 25, 1),
             "Long commute": 15 * min(e.km / 30, 1), "Little time off": 15 * (1 - min(e.leave / 5, 1)),
             "How you feel": 15 * (1 - (pulse - 1) / 4)}
    return sum(parts.values()), parts


def wellbeing_employee(e):
    st.subheader("Your wellbeing 💚")
    st.caption("Only you can see this page. HR only ever sees team-level summaries, never your score.")
    st.markdown("##### One-tap check-in: how's your workload this week?")
    cur, opts = S.pulse.get(e.id), list(PULSE)
    pick = st.radio("Pulse", opts, horizontal=True, index=opts.index(cur) if cur else None,
                    key=f"pulse_{e.id}", label_visibility="collapsed")
    if pick and pick != cur:
        if cur is None:
            S.bonus[e.id] = S.bonus.get(e.id, 0) + 5
        S.pulse[e.id] = pick
    score, parts = strain(e)
    level = "Smooth sailing 🌤️" if score < 35 else "Keep an eye on it 🌥️" if score < 60 else "Time to talk to someone 🌧️"
    c1, c2 = st.columns([1, 2])
    c1.metric("Workload strain", f"{score:.0f} / 100", help="Lower is better. Built from simple, visible factors, no black box.")
    c1.write(f"**{level}**")
    f = px.bar(pd.DataFrame({"Factor": list(parts), "Points": list(parts.values())}).sort_values("Points"), x="Points", y="Factor",
               orientation="h", color_discrete_sequence=["#2E9E6B"], title="What's behind it")
    f.update_layout(height=260, margin=dict(t=40, b=10)); c2.plotly_chart(f, use_container_width=True)
    st.caption(f"Based on: overtime {e.overtime} h, meetings {e.meetings} h, commute {e.km} km each way, "
               f"{e.leave} leave days taken, and your check-in. Nothing else.")
    st.markdown("##### Small things that would help most")
    for k in sorted(parts, key=parts.get, reverse=True)[:2]:
        st.write(f"• **{k}:** {TIPS[k]}")
    st.info("Want to raise something with the People team? The **Speak up** tab is private and anonymous.")


def speakup_employee():
    st.subheader("Speak up, privately 🕊️")
    st.markdown("No name, no login link, no tracking. Your message goes to the People team's inbox, "
                "**not** to the public Townhall board.")
    with st.form("speakup", clear_on_submit=True):
        cat = st.selectbox("What's it about?", CATS)
        msg = st.text_area("What's on your mind?", height=140, placeholder="Say it however you like. We read every message.")
        sent = st.form_submit_button("Send anonymously")
    if cat == "Harassment or safety":
        st.warning("Serious reports are routed to the designated committee and handled through the formal process.")
    if cat == "Burnout or wellbeing":
        st.info("If you're struggling right now, please also reach out to someone you trust or the company's support line.")
    if sent and msg.strip():
        code = secrets.token_hex(3).upper()
        S.concerns.append({"code": code, "category": cat, "text": msg.strip(), "status": "New", "reply": "", "date": date.today().strftime("%d %b")})
        S.last_code = code
    if S.get("last_code"):
        st.success("Sent. Save this code to check for a reply. We can't see who you are, so we can't recover it.")
        st.code(S.last_code)
    st.markdown("##### Have a code? Check for a reply")
    code_in = st.text_input("Your code", key="chk", placeholder="e.g. K7M2QA").strip().upper()
    if code_in:
        hit = [c for c in S.concerns if c["code"] == code_in]
        if not hit:
            st.write("No message found with that code.")
        else:
            c = hit[0]; st.write(f"Status: **{c['status']}**")
            st.write(f"Reply from the People team: {c['reply']}" if c["reply"] else "No reply yet. Check back soon.")
    st.caption("Prototype note: messages are stored with no employee ID at all, not even in the background.")


def employee_view():
    pick = st.sidebar.selectbox("Demo as", df.head(12)["name"])
    e = df[df["name"] == pick].iloc[0]
    my = trips[trips.id == e.id].set_index("day").loc[DAYS].reset_index()
    gdays, pdays = int(my.green.sum()), int((my["mode"] == "Carpool").sum())
    pts = 10 * gdays + 5 * pdays + S.bonus.get(e.id, 0)
    hero(f"Hey {e['name'].split()[0]} 👋", "Your commute already did the work. Here's what it added up to, and you didn't fill in a thing.")
    t1, tw, t2, t3, t4, ts = st.tabs(["My week", "Wellbeing", "Carpool", "Team challenge", "Townhall", "Speak up"])
    with t1:
        c = st.columns(4)
        c[0].metric("CO₂ you skipped", f"{my.saved.sum():.1f} kg", help="Compared with driving alone every day")
        c[1].metric("Green days", f"{gdays} / 5")
        c[2].metric("Points", pts)
        streak = 2 + (e.id * 3) % 9
        c[3].metric("Weekly streak", f"{streak + (gdays >= 3)} 🔥", help="A week counts with 3+ green days. Miss it and your streak pauses, it doesn't reset.")
        st.caption(f"≈ what a tree absorbs in {tree_days(my.saved.sum()):.0f} days. We pick this up from workplace data; you just show up.")
        show = my[["day", "mode", "source", "emitted", "saved"]].rename(columns={
            "mode": "How you got here", "source": "Captured from", "emitted": "kg CO₂ used", "saved": "kg CO₂ skipped"})
        st.dataframe(show.round(2), hide_index=True, use_container_width=True)
        with st.expander("Something look off? Fix a day (rarely needed)"):
            st.selectbox("Day", DAYS, key="fixday"); st.selectbox("Actually I", list(EF), key="fixmode")
            if st.button("Send correction"): st.success("Thanks! Sent to HR as an exception to review.")
    with t2:
        st.subheader("Ride buddies near you 🚗")
        cand = df[(df.zone == e.zone) & (df.id != e.id) & ((df["shift"] - e["shift"]).abs() <= .5)].copy()
        cand["gap"] = (cand.km - e.km).abs()
        cand = cand.sort_values("gap").head(3)
        st.caption("Matched on zone, shift time and distance. Nobody shares an address.")
        for r in cand.itertuples():
            a, b = st.columns([4, 1])
            a.write(f"**{r.name}** · {r.dept} · starts {int(r.shift)}:{'30' if r.shift % 1 else '00'}  <span class='chip'>{r.gap} km difference</span>", unsafe_allow_html=True)
            if b.button("Join pool", key=f"j{r.id}"):
                S.pools.setdefault(e.id, set()).add(r.id)
        mates = S.pools.get(e.id, set())
        if mates:
            occ = 1 + len(mates); saved = .17 * 2 * e.km * (1 - 1 / occ) * 3
            st.success(f"You're pooling with {len(mates)} colleague(s). Pooling 3 days a week saves about **{saved:.1f} kg CO₂** "
                       f"and a few parking headaches. Credits arrive on their own once the pool is confirmed at the gate.")
    with t3:
        theme = S.theme; prog, goal = team_progress(theme)
        st.subheader(f"This week: {theme}"); st.caption(THEMES[theme][2])
        mine = e.team; st.progress(min(prog[mine] / goal, 1.0), text=f"Your team ({mine}): {prog[mine]} / {goal}")
        fig = px.bar(prog.reset_index().rename(columns={0: "trips", "team": "Team"}), x="Team", y="trips",
                     color_discrete_sequence=["#2E9E6B"]); fig.add_hline(y=goal, line_dash="dot")
        fig.update_layout(height=280, margin=dict(t=10, b=10)); st.plotly_chart(fig, use_container_width=True)
        st.caption("Challenges rotate weekly so it never gets stale. Team goals, not individual rankings.")
        rot = list(THEMES); i = rot.index(theme)
        st.write("Coming up: " + "  →  ".join(rot[(i + k) % 4] for k in range(1, 4)))
    with t4:
        townhall_employee(e)
    with tw:
        wellbeing_employee(e)
    with ts:
        speakup_employee()


# ---------------- HR view ----------------
def hr_view():
    hero("People & Planet dashboard", "One score for how the workplace is doing. Team-level only: no individual is ever named.")
    tabs = st.tabs(["Sustainability Index", "Wellbeing risk", "Fairness check", "Green data", "Challenges & nudges", "Townhall", "Concerns inbox"])
    comp = wsi(); weights = {"Wellbeing": .35, "Green participation": .30, "Fairness": .15, "Voice (Townhall)": .20}
    score = sum(comp[k] * w for k, w in weights.items())
    with tabs[0]:
        c1, c2 = st.columns([1, 2])
        g = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": "Workforce Sustainability Index"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2E9E6B"}}))
        g.update_layout(height=280, margin=dict(t=50, b=0)); c1.plotly_chart(g, use_container_width=True)
        with c2:
            m = st.columns(4)
            for col, (k, v) in zip(m, comp.items()): col.metric(k, f"{v:.0f}")
            wk = [f"W{i}" for i in range(1, 13)]
            rng = np.random.default_rng(3); tr = score - np.linspace(9, 0, 12) + rng.normal(0, 1, 12)
            f = px.line(x=wk, y=tr, markers=True, labels={"x": "", "y": "Index"}, color_discrete_sequence=["#2E9E6B"])
            f.update_layout(height=220, margin=dict(t=10, b=10)); c2.plotly_chart(f, use_container_width=True)
            st.caption("Trend history is simulated in the demo.")
        d = df.assign(green=df.id.map(trips.groupby("id").green.mean())).groupby("dept").agg(
            Wellbeing=("risk", lambda x: (1 - x.mean()) * 100), Green=("green", lambda x: x.mean() * 100)).round(0).reset_index()
        st.plotly_chart(px.bar(d.melt("dept"), x="dept", y="value", color="variable", barmode="group",
                               color_discrete_sequence=["#2E9E6B", "#9AD3B5"], labels={"value": "Score", "dept": ""}), use_container_width=True)
        st.caption("Weights: Wellbeing 35%, Green 30%, Voice 20%, Fairness 15%. Adjustable in the real product.")
    with tabs[1]:
        S.thr = st.slider("Flag a person as 'at risk' above", .3, .9, S.thr, .05)
        d = df.assign(flag=df.risk >= S.thr)
        contrib = pd.DataFrame(Xs * coef, columns=list(FEATS)); contrib["team"] = df.team.values
        top = contrib.groupby("team").mean().idxmax(axis=1)
        tt = d.groupby("team").agg(People=("id", "size"), AtRisk=("flag", "sum"), AvgRisk=("risk", "mean")).reset_index()
        tt["Main driver"] = tt.team.map(top).map(FEATS); tt["Suggested action"] = tt.team.map(top).map(FIX)
        tt = tt[tt.People >= 5].sort_values("AvgRisk", ascending=False); tt["AvgRisk"] = (tt.AvgRisk * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(tt.rename(columns={"team": "Team", "AtRisk": "At risk"}), hide_index=True, use_container_width=True)
        imp = pd.DataFrame({"Factor": [FEATS[k] for k in FEATS], "Weight": coef}).sort_values("Weight")
        st.plotly_chart(px.bar(imp, x="Weight", y="Factor", orientation="h", color_discrete_sequence=["#2E9E6B"],
                               title="What the model leans on (positive = raises risk)"), use_container_width=True)
        st.caption("Gender and age are never inputs to the model. Only teams of 5+ are shown.")
    with tabs[2]:
        res, gap = fairness(S.thr)
        st.metric("Biggest gap in flag rate between groups", f"{gap:.1f} pts", delta="Looks balanced" if gap <= 10 else "Worth a closer look", delta_color="normal" if gap <= 10 else "inverse")
        a, b = st.columns(2)
        for col, (k, s) in zip([a, b], res.items()):
            col.plotly_chart(px.bar(s.reset_index(), x=k, y="flag", labels={"flag": "% flagged", k: ""}, color_discrete_sequence=["#2E9E6B"],
                                    title=f"Flag rate by {k.replace('_', ' ')}"), use_container_width=True)
        st.caption("Rule of thumb: gaps over ~10 points get reviewed. A gap can mean real workload differences or model bias, so a human checks before acting.")
    with tabs[3]:
        st.markdown("**How data arrives (no manual entry):**")
        st.dataframe(pd.DataFrame({"Source": ["HRMS attendance", "Parking-gate / vehicle tags", "Shuttle swipes", "Transit-pass taps", "Carpool confirmations"],
                                   "Gives us": ["WFH days", "Solo vs pooled arrivals", "Bus commutes", "Metro/train commutes", "Pool size per trip"],
                                   "Employee effort": ["None"] * 4 + ["One tap to join"]}), hide_index=True, use_container_width=True)
        m = st.columns(3)
        m[0].metric("CO₂ avoided this week", f"{trips.saved.sum() / 1000:.2f} t"); m[1].metric("Green commute share", f"{trips.green.mean() * 100:.0f}%")
        m[2].metric("Pooled trips", int((trips['mode'] == 'Carpool').sum()))
        e = trips.groupby("mode").emitted.sum().reset_index()
        st.plotly_chart(px.bar(e, x="mode", y="emitted", color_discrete_sequence=["#2E9E6B"], labels={"emitted": "kg CO₂", "mode": ""}), use_container_width=True)
        st.caption("Emission factors are approximate averages per km. Demo data is simulated.")
    with tabs[4]:
        S.theme = st.selectbox("This week's theme", list(THEMES), index=list(THEMES).index(S.theme))
        S.nudge_cap = st.slider("Max nudges per employee per week", 0, 5, S.nudge_cap)
        st.write("Rotation: " + "  →  ".join(THEMES))
        st.info("Anti-fatigue rules: themes rotate weekly, streaks forgive a miss, rewards go to teams, and nobody gets more than the nudge cap.")
    with tabs[5]:
        st.subheader("Townhall prep 🎤")
        qs = sorted(S.questions, key=lambda x: -x["votes"])
        st.markdown("**Suggested agenda (top voted)**")
        for i, q in enumerate(qs[:5], 1):
            st.write(f"{i}. {q['q']}  <span class='chip'>{q['votes']} votes</span> " + "".join(f"<span class='chip'>{t}</span>" for t in tag(q["q"])), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        th = pd.Series([t for q in S.questions for t in tag(q["q"]) for _ in range(q["votes"])]).value_counts().reset_index()
        th.columns = ["Theme", "Weighted votes"]
        c1.plotly_chart(px.bar(th, x="Theme", y="Weighted votes", color_discrete_sequence=["#2E9E6B"], title="What people care about"), use_container_width=True)
        c2.plotly_chart(px.bar(df.groupby("dept").pulse.mean().round(2).reset_index(), x="dept", y="pulse", color_discrete_sequence=["#9AD3B5"],
                               labels={"pulse": "Avg pulse (1-5)", "dept": ""}, title="Pulse by department"), use_container_width=True)
        st.caption("Theme tagging is keyword-based in this prototype; a language model can replace it later.")
        st.markdown("**Follow-through tracker** (shown to employees as 'You said, we did')")
        for a in S.actions:
            a["status"] = st.selectbox(f"{a['said']} → {a['did']}", ["Planned", "In progress", "Done"], index=["Planned", "In progress", "Done"].index(a["status"]), key=a["said"])
    with tabs[6]:
        st.subheader("Concerns inbox 📥")
        cs, order = S.concerns, {"New": 0, "In review": 1, "Resolved": 2}
        m = st.columns(3)
        for col, s in zip(m, order): col.metric(s, sum(c["status"] == s for c in cs))
        flt = st.multiselect("Categories", CATS, default=CATS)
        cnt = pd.Series([c["category"] for c in cs]).value_counts().reset_index(); cnt.columns = ["Category", "Messages"]
        st.plotly_chart(px.bar(cnt, x="Category", y="Messages", color_discrete_sequence=["#2E9E6B"]), use_container_width=True)
        shown = sorted([c for c in cs if c["category"] in flt], key=lambda c: (c["category"] != "Harassment or safety", order[c["status"]]))
        for c in shown:
            icon = "🔴" if c["category"] == "Harassment or safety" else "🟢" if c["status"] == "Resolved" else "🟡"
            with st.expander(f"{icon} {c['category']} · {c['date']} · {c['status']}"):
                st.write(c["text"])
                c["status"] = st.selectbox("Status", list(order), index=order[c["status"]], key=f"st{c['code']}")
                c["reply"] = st.text_area("Reply (the sender sees it only by entering their code)", c["reply"], key=f"rp{c['code']}")
        st.caption("No sender information exists, so nobody can be identified. Safety reports are pinned to the top.")


def hero(title, sub):
    st.markdown(f"<div class='hero'><h2>{title}</h2><p>{sub}</p></div>", unsafe_allow_html=True)


st.sidebar.title(f"🌿 {APP_NAME}")
view = st.sidebar.radio("I'm viewing as", ["Employee", "HR / People team"])
st.sidebar.caption("Prototype with simulated data. No real employee information.")
employee_view() if view == "Employee" else hr_view()
