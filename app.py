import json
import os
import toml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.metrics import confusion_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claims Assignment Analytics",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Secrets helper ────────────────────────────────────────────────────────────
# On Streamlit Cloud secrets come via st.secrets (dashboard → Secrets).
# Locally they live in .streamlit/.secrets.toml — toml.load is the fallback.
def _req_secret(key: str) -> str:
    # Cloud path: st.secrets populated from the dashboard
    try:
        val = st.secrets[key]
        if val and str(val).strip():
            return str(val).strip()
    except (KeyError, FileNotFoundError):
        pass
    # Local path: .streamlit/.secrets.toml
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", ".secrets.toml")
    try:
        cfg = toml.load(local_path)
        if key in cfg and str(cfg[key]).strip():
            return str(cfg[key]).strip()
    except FileNotFoundError:
        pass
    raise RuntimeError(f"Missing secret: {key}")

# ── LangChain LLM client ──────────────────────────────────────────────────────
@st.cache_resource
def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        temperature=0,
        azure_endpoint=_req_secret("AZURE_CHAT_ENDPOINT"),
        openai_api_key=_req_secret("AZURE_CHAT_API_KEY"),
        openai_api_version=_req_secret("AZURE_CHAT_API_VERSION"),
        deployment_name=_req_secret("AZURE_CHAT_DEPLOYMENT"),
        model_name=_req_secret("AZURE_CHAT_MODEL"),
        model_kwargs={"response_format": {"type": "json_object"}},
    )

# ── Load pilot data ───────────────────────────────────────────────────────────
@st.cache_data
def load_pilot_data():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot", "data")
    def r(f): return pd.read_csv(os.path.join(base, f))
    return {
        "claims_2024": r("cp_claims_2024.csv"),
        "history":     r("cp_assignment_history.csv"),
        "adj_perf":    r("cp_adjuster_performance.csv"),
        "step1":       r("step1_tagged_assignments.csv"),
        "step2":       r("step2_feature_matrix.csv"),
        "step3":       r("step3_clusters.csv"),
        "step4":       r("step4_predictions.csv"),
        "step5":       r("step5_scorer_results.csv"),
        "step6":       r("step6_llm_classifications.csv"),
        "step10":      r("step10_shadow_results.csv"),
    }

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    frames = []
    for fname in [
        "table1_claim_exposure_data.csv",
        "table2_assignment_data.csv",
        "table3_adjuster_profile_capability.csv",
        "table4_adjuster_capacity.csv",
    ]:
        df = pd.read_csv(os.path.join(base, fname))
        df.fillna("", inplace=True)
        frames.append(df)
    return frames

# ── LLM: parse all activity notes for one claim in a single call ──────────────
@st.cache_data(show_spinner=False)
def parse_claim_journey(claim_num: str, notes_json: str) -> list[dict]:
    """
    Sends all assignment events for a claim to GPT-4o in one call.
    Returns a list of {"event": N, "assigned_to": "...", "reason": "..."}.
    notes_json is a JSON string so it is hashable for @st.cache_data.
    """
    events = json.loads(notes_json)
    events_text = "\n".join(
        f"Event {e['num']} [{e['type']}]: {e['notes']}" for e in events
    )

    system_msg = (
        "You are an insurance claims data analyst. "
        "Parse raw assignment activity notes and extract structured information. "
        "Always respond with valid JSON. Be concise and precise. "
        "Never fabricate details not present in the notes."
    )
    user_msg = f"""Parse the following assignment events for claim {claim_num}.

For each event extract:
1. "assigned_to" — Full name of the adjuster who received the claim at this step.
   Look for: "assigned to [Name]", "routed to [Name]", "transferred to [Name]",
   "escalated to [Name]", "referred to [Name]", etc.
2. "reason" — Concise 1-sentence reason explaining WHY this assignment happened,
   inferred from the action/flag described.

Events:
{events_text}

Respond ONLY as a JSON object with a single key "results" containing an array — one object per event:
{{"results": [{{"event": 1, "assigned_to": "Full Name", "reason": "concise reason"}}, ...]}}"""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ])

    parsed = json.loads(response.content)
    # Handle both {"results": [...]} and bare [...] responses
    if isinstance(parsed, list):
        return parsed
    for key in ("results", "events", "data", "assignments"):
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []

# ── UI helpers ────────────────────────────────────────────────────────────────
EVENT_COLORS = {
    "Assigned":   "#1f77b4",
    "Reassigned": "#ff7f0e",
    "Escalated":  "#d62728",
    "Referred":   "#9467bd",
}

def badge(label: str) -> str:
    color = EVENT_COLORS.get(label, "#666")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;">{label}</span>'
    )

def info_box(content: str, border_color: str = "#aaa") -> str:
    return (
        f'<div style="background:#1e2535;padding:10px 14px;'
        f'border-left:3px solid {border_color};border-radius:4px;'
        f'font-size:0.87rem;line-height:1.6;">{content}</div>'
    )

# ── Pilot UI helpers ───────────────────────────────────────────────────────────
PALETTE = {"0": "#2ecc71", "A": "#3498db", "B": "#e67e22", "C": "#e74c3c"}

def step_header(num: str, title: str, description: str, color: str = "#3498db"):
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{color}22,{color}11);'
        f'border-left:4px solid {color};border-radius:8px;padding:16px 20px;margin-bottom:16px">'
        f'<div style="font-size:0.72rem;font-weight:700;color:{color};letter-spacing:.08em;'
        f'text-transform:uppercase">{num}</div>'
        f'<div style="font-size:1.35rem;font-weight:700;margin:2px 0 6px">{title}</div>'
        f'<div style="font-size:0.88rem;color:#bbb">{description}</div></div>',
        unsafe_allow_html=True,
    )

def io_card(label: str, items: list, color: str = "#555"):
    bullets = "".join(f"<li style='margin:3px 0'>{i}</li>" for i in items)
    st.markdown(
        f'<div style="background:#1a1f2e;border:1px solid {color}44;border-radius:8px;'
        f'padding:12px 16px">'
        f'<div style="font-size:0.7rem;font-weight:700;color:{color};text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:8px">{label}</div>'
        f'<ul style="margin:0;padding-left:18px;font-size:0.85rem;color:#ccc">{bullets}</ul></div>',
        unsafe_allow_html=True,
    )

# ── Sidebar navigation ────────────────────────────────────────────────────────
PILOT_STEPS = [
    "📌  Overview — The Problem",
    "1 · Tag Failure Causes",
    "2 · Build Feature Matrix",
    "3 · Cluster Failure Types",
    "4 · FNOL Complexity Model",
    "5 · Adjuster Scorer",
    "6 · Note Classification",
    "9 · KPI Dashboard",
    "10 · Shadow Pilot",
    "✅  Results & Next Steps",
]

with st.sidebar:
    st.title("📋 Claims Assignment")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Data Explorer", "Claim Journey Analyzer", "Pilot Results"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if page == "Pilot Results":
        st.markdown("**Pilot Step**")
        pilot_step = st.radio("Pilot Step", PILOT_STEPS, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Steps 7 & 8 excluded —\nIT / Compliance required.")

# ── Load all tables ───────────────────────────────────────────────────────────
t1, t2, t3, t4 = load_data()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
if page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("Preview of the four mock data tables powering the demo.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Claim & Exposure Data",
        "🔄 Assignment Data",
        "👤 Adjuster Profiles",
        "⚖️ Adjuster Capacity",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Table 1 — Claim and Exposure Data")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(t1))
        c2.metric("Unique Claims", t1["Claim Number"].nunique())
        c3.metric("Lines of Business", t1["Line of Business"].nunique())
        st.markdown("---")
        lob_filter = st.multiselect(
            "Filter by Line of Business",
            options=sorted(t1["Line of Business"].unique()),
        )
        view = t1[t1["Line of Business"].isin(lob_filter)] if lob_filter else t1
        st.dataframe(view, use_container_width=True, height=420)

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Table 2 — Assignment Data")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Events", len(t2))
        for col, etype in zip([c2, c3, c4, c5], ["Assigned", "Reassigned", "Escalated", "Referred"]):
            col.metric(etype, len(t2[t2["Event Type"] == etype]))
        st.markdown("---")
        st.caption(
            "`Assigned To Adjuster` and `Reason for Reassignment` columns are blank — "
            "they get filled automatically by parsing the `Activity Notes` text."
        )
        event_filter = st.multiselect(
            "Filter by Event Type",
            options=["Assigned", "Reassigned", "Escalated", "Referred"],
        )
        view = t2[t2["Event Type"].isin(event_filter)] if event_filter else t2
        st.dataframe(view, use_container_width=True, height=420)

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Table 3 — Adjuster Profile and Capability")
        st.metric("Adjusters", len(t3))
        st.dataframe(t3, use_container_width=True, height=540)

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Table 4 — Adjuster Capacity")

        avail_counts = t4["Availability Status"].value_counts()
        cols = st.columns(len(avail_counts))
        for col, (status, count) in zip(cols, avail_counts.items()):
            col.metric(status, count)
        st.markdown("---")

        def highlight_avail(val):
            styles = {
                "Available":         "background-color:#1a472a;color:#90ee90",
                "Near Capacity":     "background-color:#7b5800;color:#ffd700",
                "At Capacity":       "background-color:#7b1a1a;color:#ffaaaa",
                "Overloaded":        "background-color:#4a0000;color:#ff6666",
                "Unavailable - PTO": "background-color:#2a2a5a;color:#aaaaff",
            }
            return styles.get(val, "")

        st.dataframe(
            t4.style.map(highlight_avail, subset=["Availability Status"]),
            use_container_width=True,
            height=540,
        )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLAIM JOURNEY ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Claim Journey Analyzer":
    st.title("Claim Journey Analyzer")
    st.markdown(
        "Select a claim to see its full assignment path — "
        "who handled it at each step and why, pulled directly from the activity notes."
    )

    # ── Claim selector + metadata ─────────────────────────────────────────────
    all_claims = sorted(t2["Claim Number"].unique())
    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        selected_claim = st.selectbox("Select Claim", all_claims)

    claim_meta = t1[t1["Claim Number"] == selected_claim].iloc[0]

    with col_info:
        st.markdown(
            f"**LOB:** {claim_meta['Line of Business']} &nbsp;|&nbsp; "
            f"**Type:** {claim_meta['Claim Type']} &nbsp;|&nbsp; "
            f"**State:** {claim_meta['Jurisdiction/State']} &nbsp;|&nbsp; "
            f"**Complexity:** `{claim_meta['Complexity']}` &nbsp;|&nbsp; "
            f"**FNOL:** {claim_meta['FNOL Date/Date Reported']}"
        )
        flags = []
        if claim_meta.get("Litigation Flag") == "Y":
            flags.append("⚖️ Litigation")
        if claim_meta.get("CAT Flag") == "Y":
            flags.append("🌪️ CAT")
        if claim_meta.get("Subrogation Flag") == "Y":
            flags.append("🔁 Subrogation")
        if claim_meta.get("Claim Close Date"):
            flags.append(f"✅ Closed {claim_meta['Claim Close Date']}")
        if claim_meta.get("Reopen Date"):
            flags.append(f"🔓 Reopened {claim_meta['Reopen Date']}")
        if flags:
            st.markdown("  ".join(flags))

    st.markdown("---")

    # ── Events for this claim ─────────────────────────────────────────────────
    events_df = (
        t2[t2["Claim Number"] == selected_claim]
        .sort_values("Assignment #")
        .reset_index(drop=True)
    )

    if events_df.empty:
        st.warning("No assignment events found for this claim.")
        st.stop()

    # ── Single LLM call for all events ───────────────────────────────────────
    notes_payload = [
        {"num": int(r["Assignment #"]), "type": r["Event Type"], "notes": r["Activity Notes"]}
        for _, r in events_df.iterrows()
    ]
    notes_json = json.dumps(notes_payload)

    with st.spinner(f"Analyzing {len(events_df)} events for {selected_claim}…"):
        try:
            llm_results = parse_claim_journey(selected_claim, notes_json)
            llm_map = {item.get("event", i + 1): item for i, item in enumerate(llm_results)}
            llm_error = None
        except Exception as exc:
            llm_map = {}
            llm_error = str(exc)

    if llm_error:
        st.error(f"Analysis error: {llm_error}")

    # ── Journey timeline ──────────────────────────────────────────────────────
    st.subheader(f"Assignment Journey — {selected_claim}")
    st.caption(f"{len(events_df)} events · Complexity: {claim_meta['Complexity']}")

    for _, row in events_df.iterrows():
        ev_num  = int(row["Assignment #"])
        ev_type = row["Event Type"]
        ev_data = llm_map.get(ev_num, {})
        assigned_to = ev_data.get("assigned_to", "—")
        reason      = ev_data.get("reason", "—")
        color = EVENT_COLORS.get(ev_type, "#555")

        left_col, right_col = st.columns([0.02, 0.98])
        with left_col:
            # Vertical timeline bar
            st.markdown(
                f'<div style="width:4px;min-height:130px;background:{color};'
                f'border-radius:4px;margin:0 auto;"></div>',
                unsafe_allow_html=True,
            )
        with right_col:
            with st.expander(
                f"#{ev_num}  ·  {ev_type}  ·  {row['Timestamp']}  —  {assigned_to}",
                expanded=True,
            ):
                # Badge + header rendered inside where HTML works
                st.markdown(
                    f'{badge(ev_type)} &nbsp; <span style="font-size:0.85rem;color:#ccc;">'
                    f'🕐 {row["Timestamp"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                note_col, extract_col = st.columns(2)
                with note_col:
                    st.markdown("**Activity Notes**")
                    st.markdown(
                        info_box(row["Activity Notes"], border_color=color),
                        unsafe_allow_html=True,
                    )
                with extract_col:
                    st.markdown("**Extracted Information**")
                    st.markdown(
                        info_box(
                            f"<b>Assigned To:</b>&nbsp; {assigned_to}<br><br>"
                            f"<b>Reason:</b>&nbsp; {reason}<br><br>"
                            f"<b>Trigger:</b>&nbsp; {row['Manual/System Trigger']} "
                            f"&nbsp;|&nbsp; <b>By:</b>&nbsp; {row['Assigned By']}"
                        ),
                        unsafe_allow_html=True,
                    )

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Journey Summary")

    summary_rows = []
    for _, row in events_df.iterrows():
        ev_num  = int(row["Assignment #"])
        ev_data = llm_map.get(ev_num, {})
        summary_rows.append({
            "#":               ev_num,
            "Event Type":      row["Event Type"],
            "Timestamp":       row["Timestamp"],
            "Assigned To":     ev_data.get("assigned_to", "—"),
            "Reason":          ev_data.get("reason", "—"),
            "Trigger":         row["Manual/System Trigger"],
            "Assigned By":     row["Assigned By"],
            "Exposure ID":     row["Exposure ID"],
        })

    summary_df = pd.DataFrame(summary_rows)

    def color_event_type(val):
        c = EVENT_COLORS.get(val, "")
        return f"color:{c};font-weight:600" if c else ""

    st.dataframe(
        summary_df.style.map(color_event_type, subset=["Event Type"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events",   len(events_df))
    c2.metric("Reassignments",  len(events_df[events_df["Event Type"] == "Reassigned"]))
    c3.metric("Escalations",    len(events_df[events_df["Event Type"] == "Escalated"]))
    c4.metric("Referrals",      len(events_df[events_df["Event Type"] == "Referred"]))

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PILOT RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Pilot Results":
    d = load_pilot_data()
    fm     = d["step2"]
    preds  = d["step4"]
    scores = d["step5"]
    shadow = d["step10"]
    llm    = d["step6"]

    non_stp      = fm[fm["stp_flag"] == "N"]
    ftr_rate     = (non_stp["final_group"] == "0").mean()
    group_c_rate = (non_stp["final_group"] == "C").mean()
    model_acc    = (preds["predicted_group"] == preds["final_group"]).mean()
    scorer_match = scores["top_match_actual"].mean()
    avoidable    = (llm["category"] == "A").mean()
    shadow_top1  = shadow["top1_match"].mean()
    shadow_top3  = shadow["top3_match"].mean()

    # ── Overview ──────────────────────────────────────────────────────────────
    if pilot_step == PILOT_STEPS[0]:
        st.title("Commercial Property — Reassignment Pilot")
        st.markdown("**Why are 77% of non-STP claims failing to be resolved by the first adjuster?**")
        st.markdown("---")

        st.subheader("Current State Baseline")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CP Claims (2024)",       "200")
        c2.metric("Non-STP Claims",         f"{len(non_stp)}")
        c3.metric("First-Touch Resolution", f"{ftr_rate*100:.0f}%",
                  delta="Target: 35%+", delta_color="inverse")
        c4.metric("Group C Rate",           f"{group_c_rate*100:.0f}%",
                  delta="Highest complexity bucket", delta_color="off")
        c5.metric("Avg Assignments/Claim",  f"{fm['assignment_count'].mean():.1f}")
        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            grp = fm["final_group"].value_counts().reset_index()
            grp.columns = ["Group", "Claims"]
            grp["Label"] = grp["Group"].map({
                "0": "Group 0 — Resolved 1st Touch",
                "A": "Group A — 2 assignments",
                "B": "Group B — 3–4 assignments",
                "C": "Group C — 5+ assignments",
            })
            fig = px.bar(grp.sort_values("Group"), x="Label", y="Claims", color="Group",
                         color_discrete_map=PALETTE, title="200 CP Claims by Complexity Group")
            fig.update_layout(showlegend=False, xaxis_title="")
            fig.update_xaxes(tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            ev = d["history"]["event_type"].value_counts().reset_index()
            ev.columns = ["Event Type", "Count"]
            fig = px.pie(ev, names="Event Type", values="Count",
                         title="1 116 Assignment Events — Type Breakdown",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("The 10-Step Pilot Pipeline")
        steps_info = [
            ("1","Tag Causes","#3498db"), ("2","Feature Matrix","#9b59b6"),
            ("3","Cluster Types","#e67e22"), ("4","FNOL Model","#2ecc71"),
            ("5","Adj. Scorer","#1abc9c"), ("6","Note Class.","#e74c3c"),
            ("9","KPI Dashboard","#f39c12"), ("10","Shadow Pilot","#34495e"),
        ]
        arrow = '<div style="font-size:1.1rem;color:#555;font-weight:700">→</div>'
        cells = "".join(
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="background:{c};color:#fff;border-radius:8px;padding:8px 12px;'
            f'text-align:center;min-width:76px;font-size:0.76rem;font-weight:700;line-height:1.4">'
            f'Step {n}<br>{t}</div>'
            f'{arrow if i < len(steps_info) - 1 else ""}'
            f'</div>'
            for i, (n, t, c) in enumerate(steps_info)
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;'
            f'background:#1a1f2e;border-radius:10px;padding:16px 20px">{cells}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.info("**Steps 7 & 8** (IT integration + Compliance sign-off) are excluded — "
                "they require production system access and formal approval.")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[1]:
        step_header("Step 1", "Tag Failure Causes",
                    "Label every reassignment event with a structured cause — turning raw "
                    "narrative notes into a categorical signal the model can learn from.", "#3498db")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["cp_assignment_history.csv — 1 116 events",
                               "Raw activity notes (free-text)", "Trigger: System / Manual",
                               "Event type: Assigned / Reassigned / Escalated / Referred"], "#3498db")
        with col_out:
            io_card("Output", ["step1_tagged_assignments.csv — 1 116 rows",
                                "cause_tag: 7 distinct categories",
                                "is_user_auto flag", "cause_tag_refined"], "#2ecc71")
        st.markdown("")

        cause = d["step1"]["cause_tag"].value_counts().reset_index()
        cause.columns = ["Cause Tag", "Events"]
        col_c, col_d = st.columns([3, 2])
        with col_c:
            fig = px.bar(cause, x="Events", y="Cause Tag", orientation="h",
                         color="Cause Tag", title="Events by Cause Tag",
                         color_discrete_sequence=px.colors.qualitative.Pastel, text="Events")
            fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        with col_d:
            st.markdown("**Cause Tag Definitions**")
            defs = {
                "Manual - Supervisor":      "Supervisor manually re-routed the claim",
                "System - Auto Route":      "Initial FNOL auto-assignment by rules engine",
                "System - Rule Triggered":  "SLA breach or exposure change triggered move",
                "Manual - Workload":        "Workload rebalance by team lead",
                "User Automated":           "Adjuster action auto-processed by system",
                "Named Account Bypass":     "Account-level preferred adjuster override",
                "Fallback Cascade":         "Original adjuster unavailable — cascaded",
            }
            for tag, defn in defs.items():
                n = cause.loc[cause["Cause Tag"] == tag, "Events"].values
                pct = round(int(n[0]) / len(d["step1"]) * 100, 1) if len(n) else 0
                st.markdown(f"**{tag}** ({pct}%): {defn}")

        st.markdown("---")
        st.subheader("Sample Tagged Events")
        st.dataframe(
            d["step1"][["claim_id","event_type","cause_tag","trigger","activity_notes"]].head(10),
            use_container_width=True, hide_index=True,
        )

    # ── Step 2 ────────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[2]:
        step_header("Step 2", "Build Feature Matrix",
                    "Collapse 1 116 raw events into one analytical row per claim — "
                    "the training surface for every downstream model.", "#9b59b6")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["step1_tagged_assignments.csv — event-level",
                               "cp_claims_2024.csv — claim metadata",
                               "FNOL date, loss state, STP flag, loss amount"], "#9b59b6")
        with col_out:
            io_card("Output", ["step2_feature_matrix.csv — 200 rows × 26 cols",
                                "assignment_count, pct_manual, pct_system",
                                "n_unique_adjusters, same_adjuster_returned",
                                "geographic_mismatch, avg_hrs_between_events"], "#2ecc71")
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.histogram(fm, x="assignment_count", color="final_group",
                               color_discrete_map=PALETTE, nbins=12, barmode="overlay",
                               opacity=0.75, title="# Assignments per Claim",
                               category_orders={"final_group": ["0","A","B","C"]})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(fm, x="pct_manual", color="final_group",
                               color_discrete_map=PALETTE, nbins=10, barmode="overlay",
                               opacity=0.75, title="% Manual Triggers per Claim",
                               category_orders={"final_group": ["0","A","B","C"]})
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            geo = fm.groupby("final_group")["geographic_mismatch"].mean().reset_index()
            geo.columns = ["Group", "Avg Geo Mismatch"]
            fig = px.bar(geo, x="Group", y="Avg Geo Mismatch", color="Group",
                         color_discrete_map=PALETTE, title="Avg Geo Mismatch by Group")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Matrix — First 10 Rows")
        display_cols = ["claim_id","final_group","assignment_count","pct_manual","pct_system",
                        "n_unique_adjusters","same_adjuster_returned","geographic_mismatch",
                        "avg_hrs_between_events"]
        st.dataframe(fm[display_cols].head(10), use_container_width=True, hide_index=True)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[3]:
        step_header("Step 3", "Cluster Failure Types",
                    "Use unsupervised clustering to identify *why* claims fail — "
                    "not just how many times they bounced.", "#e67e22")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["step2_feature_matrix.csv — 154 non-FTR claims",
                               "10 numerical features (StandardScaler)",
                               "Groups A, B, C only (FTR excluded)"], "#e67e22")
        with col_out:
            io_card("Output", ["step3_clusters.csv — 154 rows with cluster label",
                                "4 failure archetypes via KMeans k=4",
                                "Visualized in 2D with t-SNE"], "#2ecc71")
        st.markdown("")

        clusters = d["step3"]
        cc = clusters["cluster_label"].value_counts().reset_index()
        cc.columns = ["Cluster", "Claims"]
        col_pie, col_desc = st.columns([2, 3])
        with col_pie:
            fig = px.pie(cc, names="Cluster", values="Claims",
                         title="154 Non-FTR Claims — Failure Archetypes",
                         color_discrete_sequence=px.colors.qualitative.Set2, hole=0.35)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        with col_desc:
            st.markdown("**What Each Cluster Means**")
            cluster_defs = [
                ("Complexity Escalation",   "#2ecc71",
                 "Claims that legitimately needed escalation. Reassignments here are appropriate."),
                ("Geographic Mismatch",     "#3498db",
                 "System routed to an adjuster outside the loss state. Fix: enforce geo matching at FNOL."),
                ("Structural Bounce",       "#e67e22",
                 "Same-tier handoffs — adjuster eventually returned. Typically a workload spike."),
                ("Manual Re-entry / Override", "#e74c3c",
                 "High manual trigger rate. Supervisors repeatedly overrode system — signals a rules gap."),
            ]
            for label, color, desc in cluster_defs:
                n = cc.loc[cc["Cluster"] == label, "Claims"].values
                n = int(n[0]) if len(n) else 0
                st.markdown(
                    f'<div style="border-left:3px solid {color};padding:8px 12px;margin-bottom:8px;'
                    f'background:#1a1f2e;border-radius:0 6px 6px 0">'
                    f'<b style="color:{color}">{label}</b> '
                    f'<span style="color:#999;font-size:0.8rem">({n} claims)</span>'
                    f'<div style="font-size:0.84rem;color:#ccc;margin-top:2px">{desc}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        CLUSTER_FEATURES = ["assignment_count","pct_manual","pct_system","n_unique_adjusters",
                            "same_adjuster_returned","geographic_mismatch","avg_hrs_between_events"]
        profile = (fm[fm["final_group"] != "0"]
                   .merge(clusters, on="claim_id", how="left")
                   .groupby("cluster_label")[CLUSTER_FEATURES].mean().round(2).reset_index()
                   .rename(columns={"cluster_label": "Cluster"}))
        st.subheader("Cluster Feature Profiles (Mean Values)")
        st.dataframe(profile, use_container_width=True, hide_index=True)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[4]:
        step_header("Step 4", "FNOL Complexity Model",
                    "Train a classifier to predict how complex a claim will be at intake — "
                    "before any assignments happen.", "#2ecc71")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["step2 FNOL-only features (no post-assignment leakage)",
                               "loss_cause, policy_type, loss_state, fnol_hour",
                               "n_exposures_at_fnol, reported_loss_amount",
                               "Label: final_group (0 / A / B / C)"], "#2ecc71")
        with col_out:
            io_card("Output", ["step4_predictions.csv — predicted_group per claim",
                                "step4_complexity_model.pkl — trained XGBoost model",
                                "Used by Steps 5 and 10"], "#2ecc71")
        st.markdown("")

        st.metric("Overall FNOL Complexity Accuracy", f"{model_acc*100:.1f}%",
                  delta="Target: >70%",
                  delta_color="normal" if model_acc >= 0.70 else "inverse")
        st.markdown("")

        col_acc, col_conf = st.columns(2)
        with col_acc:
            acc_grp = (preds.assign(correct=preds["predicted_group"] == preds["final_group"])
                       .groupby("final_group")["correct"].agg(["sum","count"])
                       .rename(columns={"sum":"Correct","count":"Total"})
                       .assign(Accuracy=lambda x: x["Correct"] / x["Total"])
                       .reset_index().rename(columns={"final_group":"Group"}))
            fig = px.bar(acc_grp, x="Group", y="Accuracy", color="Group",
                         color_discrete_map=PALETTE, title="Accuracy by Group",
                         text=acc_grp["Accuracy"].map(lambda v: f"{v*100:.0f}%"))
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            fig.add_hline(y=0.70, line_dash="dash", line_color="#aaa",
                          annotation_text="70% target")
            st.plotly_chart(fig, use_container_width=True)
        with col_conf:
            labels_order = ["0","A","B","C"]
            cm = confusion_matrix(preds["final_group"], preds["predicted_group"],
                                  labels=labels_order)
            cm_df = pd.DataFrame(cm,
                                 index=[f"Actual {g}" for g in labels_order],
                                 columns=[f"Pred {g}" for g in labels_order])
            st.markdown("**Confusion Matrix**")
            st.dataframe(cm_df, use_container_width=True)
            st.caption("Rows = actual · Columns = predicted · Diagonal = correct")

        st.markdown("---")
        st.info("The model uses **only FNOL-time features** — allowing the system to route "
                "a claim to the right adjuster tier *before* the first assignment is made.")

    # ── Step 5 ────────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[5]:
        step_header("Step 5", "Adjuster Scorer",
                    "Score every eligible adjuster for each incoming claim using a "
                    "weighted composite formula.", "#1abc9c")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["step4_predictions.csv — predicted complexity",
                               "cp_adjuster_performance.csv — 7 CP adjusters",
                               "historical_ftr_rate, utilisation_pct",
                               "cp_experience_years, large_loss_certified"], "#1abc9c")
        with col_out:
            io_card("Output", ["step5_scorer_results.csv — top scorer per claim",
                                "Composite score 0–1 per adjuster-claim pair",
                                "top_match_actual: 1 if top scorer = final owner"], "#2ecc71")
        st.markdown(
            '<div style="background:#1a2535;border:1px solid #1abc9c44;border-radius:8px;'
            'padding:12px 18px;margin:12px 0">'
            '<div style="font-size:0.76rem;color:#1abc9c;font-weight:700;margin-bottom:4px">'
            'SCORING FORMULA</div>'
            '<code style="font-size:0.9rem;color:#eee">'
            'Score = 0.30 × LOB_match + 0.25 × FTR_rate + 0.20 × (1 − utilisation) + 0.25 × experience_alignment'
            '</code></div>',
            unsafe_allow_html=True,
        )
        st.metric("Scorer Top-1 Match Rate", f"{scorer_match*100:.1f}%",
                  delta="Target: >50%",
                  delta_color="normal" if scorer_match >= 0.50 else "inverse")
        st.markdown("")

        col_adj, col_score = st.columns(2)
        with col_adj:
            adj = d["adj_perf"][["adjuster_name","historical_ftr_rate","utilisation_pct",
                                  "cp_experience_years","large_loss_certified"]].copy()
            adj.columns = ["Adjuster","FTR Rate","Utilisation","CP Exp (yrs)","Large Loss"]
            adj["FTR Rate"]    = adj["FTR Rate"].map(lambda v: f"{v*100:.0f}%")
            adj["Utilisation"] = adj["Utilisation"].map(lambda v: f"{v*100:.0f}%")
            st.subheader("Adjuster Performance Profile")
            st.dataframe(adj, use_container_width=True, hide_index=True)
        with col_score:
            top_c = (scores.groupby("top_scorer_name")
                     .agg(Recommended=("claim_id","count"), Avg_Score=("top_score","mean"))
                     .round(3).reset_index().sort_values("Recommended", ascending=False)
                     .rename(columns={"top_scorer_name":"Adjuster","Avg_Score":"Avg Score"}))
            fig = px.bar(top_c, x="Adjuster", y="Recommended", color="Avg Score",
                         color_continuous_scale="Teal",
                         title="Top Recommendation Frequency", text="Recommended")
            fig.update_layout(xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    # ── Step 6 ────────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[6]:
        step_header("Step 6", "Reassignment Note Classification",
                    "Use an LLM to classify each reassignment note as avoidable, necessary, "
                    "manual override, or unclear.", "#e74c3c")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["50 sampled activity notes (free text)",
                               "Drawn from non-FTR reassignment / escalation events",
                               "Sent in batches of 10"], "#e74c3c")
        with col_out:
            io_card("Output", ["step6_llm_classifications.csv — 50 rows",
                                "category: A / B / C / D",
                                "Feeds into Step 9 KPI dashboard"], "#2ecc71")
        st.markdown("")

        for label, color, desc in [
            ("A — Avoidable",       "#e74c3c", "Better routing or rule could have prevented this"),
            ("B — Necessary",       "#3498db", "Genuine complexity required a specialist"),
            ("C — Manual Override", "#e67e22", "Adjuster/supervisor bypassed system recommendation"),
            ("D — Unclear",         "#888",    "Notes too vague to classify with confidence"),
        ]:
            st.markdown(
                f'<span style="background:{color}22;border:1px solid {color}55;border-radius:6px;'
                f'padding:5px 10px;margin:2px 4px;display:inline-block;font-size:0.84rem">'
                f'<b style="color:{color}">{label}</b>: {desc}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("")

        cat_counts = llm["category"].value_counts().reset_index()
        cat_counts.columns = ["Category","Count"]
        cat_colors = {"A":"#e74c3c","B":"#3498db","C":"#e67e22","D":"#888"}
        col_pie, col_kpi = st.columns(2)
        with col_pie:
            fig = px.pie(cat_counts, names="Category", values="Count", color="Category",
                         color_discrete_map=cat_colors,
                         title="50 Sampled Notes — Classification", hole=0.4)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        with col_kpi:
            av_n   = int(cat_counts.loc[cat_counts["Category"] == "A", "Count"].values[0]) \
                     if "A" in cat_counts["Category"].values else 0
            av_pct = av_n / len(llm) * 100
            st.metric("Avoidable Reassignments", f"{av_pct:.0f}%",
                      delta=f"{av_n} of 50 sampled events", delta_color="off")
            st.markdown(
                f"If **{av_pct:.0f}%** of reassignments are avoidable, this translates to roughly "
                f"**{fm['assignment_count'].mean() * len(fm) * av_pct / 100:.0f} unnecessary "
                "handoffs** across 200 claims that better routing could eliminate."
            )
        st.markdown("---")
        st.caption("⚠️ step6_llm_classifications.csv uses mock data in this demo. "
                   "Run step6_llm_parsing.ipynb with a live API key for real output.")

    # ── Step 9 ────────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[7]:
        step_header("Step 9", "Feedback Loop & KPI Dashboard",
                    "Aggregate all pilot outputs into a monitoring dashboard — "
                    "track progress against targets and detect model drift.", "#f39c12")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["step2 (claims + FTR labels)", "step4 (model accuracy)",
                               "step5 (match rate)", "step6 (avoidable %)"], "#f39c12")
        with col_out:
            io_card("Output", ["KPI dashboard vs targets",
                                "Weekly FTR trend chart", "Assignment distribution by group"], "#2ecc71")
        st.markdown("")

        kpi_data = [
            ("FTR Rate (non-STP)",       ftr_rate,     0.35, True,  "23% baseline → 35% target"),
            ("Group C Rate",             group_c_rate, 0.60, False, "Lower is better; target <60%"),
            ("FNOL Complexity Accuracy", model_acc,    0.70, True,  "XGBoost 4-class classifier"),
            ("Scorer Top-1 Match",       scorer_match, 0.50, True,  "Right adjuster, 1st time"),
            ("Avoidable Reassignments",  avoidable,    0.40, False, "LLM-classified (mock data)"),
        ]
        cols = st.columns(len(kpi_data))
        for col, (name, val, target, higher, note) in zip(cols, kpi_data):
            on = val >= target if higher else val <= target
            col.metric(name, f"{val*100:.1f}%",
                       delta=f"Target: {'≥' if higher else '<'}{target*100:.0f}%",
                       delta_color="normal" if on else "inverse")
            col.caption(note)
        st.markdown("")

        kpi_df = pd.DataFrame([
            {"Metric": nm.replace(" (non-STP)",""),
             "Current": v * 100, "Target": t * 100,
             "On Target": v >= t if h else v <= t}
            for nm, v, t, h, _ in kpi_data
        ])
        fig = go.Figure()
        fig.add_bar(name="Current", x=kpi_df["Metric"], y=kpi_df["Current"],
                    marker_color=["#2ecc71" if x else "#e74c3c" for x in kpi_df["On Target"]],
                    text=kpi_df["Current"].map(lambda v: f"{v:.1f}%"), textposition="outside")
        fig.add_bar(name="Target",  x=kpi_df["Metric"], y=kpi_df["Target"],
                    marker_color="rgba(255,255,255,0.15)",
                    text=kpi_df["Target"].map(lambda v: f"{v:.0f}%"),  textposition="outside")
        fig.update_layout(barmode="group", title="Pilot KPIs — Current vs Target",
                          yaxis_title="Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Weekly FTR Trend (2024 Baseline — Monitoring View)")
        hist = d["step1"].copy()
        hist["ts"] = pd.to_datetime(hist["timestamp"], format="%m/%d/%Y %I:%M %p")
        first_ev = hist.groupby("claim_id")["ts"].min().reset_index(name="first_event")
        weekly = (fm.merge(first_ev, on="claim_id")
                  .assign(week=lambda df: pd.to_datetime(df["first_event"]).dt.to_period("W").astype(str))
                  .groupby("week")
                  .apply(lambda g: (g["final_group"] == "0").mean())
                  .reset_index(name="ftr_rate"))
        fig = px.line(weekly, x="week", y="ftr_rate",
                      title="Weekly FTR Rate — 2024 Baseline",
                      labels={"week":"Week","ftr_rate":"FTR Rate"}, markers=True)
        fig.add_hline(y=0.35, line_dash="dash", line_color="#2ecc71",
                      annotation_text="Pilot Target: 35%")
        fig.add_hline(y=ftr_rate, line_dash="dot", line_color="#e67e22",
                      annotation_text=f"Baseline Avg: {ftr_rate*100:.0f}%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # ── Step 10 ───────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[8]:
        step_header("Step 10", "Shadow Pilot — 2025 Claims",
                    "Run the full pipeline on 40 brand-new claims in read-only mode: "
                    "predict complexity → score adjusters → compare vs actual final owner.", "#34495e")
        col_in, col_out = st.columns(2)
        with col_in:
            io_card("Input", ["cp_claims_2025_pilot.csv — 40 new CP claims",
                               "FNOL features only (no outcome data)",
                               "Same adjuster pool as 2024 baseline",
                               "Trained model from Step 4"], "#34495e")
        with col_out:
            io_card("Output", ["step10_shadow_results.csv — 40 rows",
                                "scorer_top1/2/3 per claim",
                                "top1_match / top3_match vs actual final adjuster"], "#2ecc71")
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        c1.metric("Shadow Claims", f"{len(shadow)}", "No live system touched", delta_color="off")
        c2.metric("Top-1 Match Rate", f"{shadow_top1*100:.1f}%", "Target: >50%",
                  delta_color="normal" if shadow_top1 >= 0.50 else "inverse")
        c3.metric("Top-3 Match Rate", f"{shadow_top3*100:.1f}%", None, delta_color="off")
        st.markdown("")

        col_grp, col_tbl = st.columns([1, 2])
        with col_grp:
            gc = shadow["predicted_group"].value_counts().reset_index()
            gc.columns = ["Group","Claims"]
            fig = px.pie(gc, names="Group", values="Claims", color="Group",
                         color_discrete_map=PALETTE, title="Predicted Complexity", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col_tbl:
            sd = shadow[["claim_id","predicted_group","scorer_top1_name","scorer_top2_name",
                         "scorer_top3_name","actual_final_adj","top1_match","top3_match"]].copy()
            sd.columns = ["Claim","Predicted","Rec #1","Rec #2","Rec #3","Actual","Top-1 ✓","Top-3 ✓"]
            sd["Top-1 ✓"] = sd["Top-1 ✓"].map({1:"✅",0:"❌"})
            sd["Top-3 ✓"] = sd["Top-3 ✓"].map({1:"✅",0:"❌"})
            st.dataframe(sd, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.info("**Shadow mode = zero risk.** Recommendations were logged but never acted on. "
                "Once top-1 match rate exceeds 50%, we activate live routing.")

    # ── Results ───────────────────────────────────────────────────────────────
    elif pilot_step == PILOT_STEPS[9]:
        st.title("Results & Next Steps")
        st.markdown("What the pilot proved, what it means for operations, and where to go next.")
        st.markdown("---")

        st.subheader("What the Pilot Proved")
        findings = [
            ("📊","#3498db","The failure rate is structural",
             f"77% of non-STP CP claims are reassigned at least once. "
             f"Group C alone is {(non_stp['final_group']=='C').mean()*100:.0f}% of non-STP volume — "
             "predictable patterns, not random failures."),
            ("🔬","#9b59b6","4 distinct failure archetypes identified",
             "KMeans clustering found Complexity Escalation, Geographic Mismatch, "
             "Structural Bounce, and Manual Override — each requiring a different fix."),
            ("🤖","#2ecc71",f"FNOL model reaches {model_acc*100:.0f}% accuracy",
             "An XGBoost classifier trained only on FNOL-time features predicts final "
             "complexity accurately enough to inform initial routing decisions."),
            ("🎯","#1abc9c",f"Adjuster scorer matches actual owner {scorer_match*100:.0f}% of the time",
             "A weighted composite scorer identifies the correct final adjuster as its "
             "top pick in most cases. Shadow pilot confirms this on fresh 2025 data."),
            ("📝","#e74c3c",f"{avoidable*100:.0f}% of sampled reassignments are avoidable",
             "LLM classification found a large share of reassignments resulted from routing "
             "mismatches or workload issues — not genuine complexity."),
        ]
        for icon, color, title, body in findings:
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:10px 16px;margin-bottom:8px;'
                f'background:#1a1f2e;border-radius:0 8px 8px 0">'
                f'<span style="font-size:1.1rem">{icon}</span>'
                f' <b style="color:{color}">{title}</b>'
                f'<div style="font-size:0.87rem;color:#ccc;margin-top:3px">{body}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("Baseline vs Pilot Target")
        total = len(non_stp)
        base_n, tgt_n = int(total * 0.23), int(total * 0.35)
        cl, _, cr = st.columns([5,1,5])
        with cl:
            st.markdown(
                f'<div style="background:#7b1a1a;border-radius:10px;padding:20px;text-align:center">'
                f'<div style="font-size:2.5rem;font-weight:800;color:#ff6666">23%</div>'
                f'<div style="font-size:1rem;color:#ffaaaa">Current FTR Rate</div>'
                f'<div style="font-size:0.84rem;color:#cc8888;margin-top:6px">'
                f'{base_n} of {total} non-STP claims resolved first touch</div></div>',
                unsafe_allow_html=True,
            )
        with _:
            st.markdown("<div style='text-align:center;font-size:2rem;margin-top:26px'>→</div>",
                        unsafe_allow_html=True)
        with cr:
            st.markdown(
                f'<div style="background:#1a472a;border-radius:10px;padding:20px;text-align:center">'
                f'<div style="font-size:2.5rem;font-weight:800;color:#90ee90">35%+</div>'
                f'<div style="font-size:1rem;color:#aaffaa">Pilot Target FTR Rate</div>'
                f'<div style="font-size:0.84rem;color:#88cc88;margin-top:6px">'
                f'{tgt_n} of {total} claims — +{tgt_n - base_n} fewer bounced claims</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.markdown(
            f"At 35% FTR, **{tgt_n - base_n} additional claims** per 200 would be resolved "
            "by the first adjuster — each avoided reassignment saves handler time, "
            "reduces cycle days, and improves customer experience."
        )

        st.markdown("---")
        st.subheader("Recommended Next Steps")
        for num, title, color, body in [
            ("Step 7",  "IT Integration",      "#e67e22",
             "Connect the scoring engine to the claims management system via API. "
             "Surface top-3 adjuster recommendations at FNOL screen — one click to accept."),
            ("Step 8",  "Compliance Sign-off",  "#e74c3c",
             "Submit the model card, bias analysis, and audit trail to Legal & Compliance. "
             "Required before live routing can be enabled."),
            ("Ongoing", "Model Monitoring",     "#3498db",
             "Re-run Step 9 KPI dashboard weekly. Retrain the FNOL model quarterly "
             "or when drift is detected (accuracy drops >5pp below baseline)."),
            ("Scale",   "Expand to Other LOBs", "#9b59b6",
             "Once CP FTR target is met for 3 consecutive months, apply the same "
             "pipeline to Workers Comp and General Liability."),
        ]:
            st.markdown(
                f'<div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;'
                f'background:#1a1f2e;border-radius:8px;padding:12px 16px">'
                f'<div style="background:{color};color:#fff;border-radius:6px;padding:4px 8px;'
                f'font-size:0.76rem;font-weight:700;white-space:nowrap;min-width:52px;text-align:center">'
                f'{num}</div>'
                f'<div><b style="color:{color}">{title}</b>'
                f'<div style="font-size:0.86rem;color:#ccc;margin-top:2px">{body}</div></div></div>',
                unsafe_allow_html=True,
            )
