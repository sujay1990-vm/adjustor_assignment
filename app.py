import json
import os
import toml
import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claims Assignment Analytics",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Secrets helper ────────────────────────────────────────────────────────────
# Streamlit's st.secrets reads .streamlit/secrets.toml (no dot prefix).
# Our file is .streamlit/.secrets.toml, so we load it manually and expose the
# same _req_secret() interface.
@st.cache_resource
def _load_secrets() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", ".secrets.toml")
    return toml.load(path)

def _req_secret(key: str) -> str:
    cfg = _load_secrets()
    if key not in cfg or not str(cfg[key]).strip():
        raise RuntimeError(f"Missing secret: {key}")
    return str(cfg[key]).strip()

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

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📋 Claims Assignment")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Data Explorer", "Claim Journey Analyzer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

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
            t4.style.applymap(highlight_avail, subset=["Availability Status"]),
            use_container_width=True,
            height=540,
        )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLAIM JOURNEY ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
else:
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
        summary_df.style.applymap(color_event_type, subset=["Event Type"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events",   len(events_df))
    c2.metric("Reassignments",  len(events_df[events_df["Event Type"] == "Reassigned"]))
    c3.metric("Escalations",    len(events_df[events_df["Event Type"] == "Escalated"]))
    c4.metric("Referrals",      len(events_df[events_df["Event Type"] == "Referred"]))
