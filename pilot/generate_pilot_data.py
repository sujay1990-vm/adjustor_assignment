"""
Generate all mock data for the Commercial Property pilot.

Metrics from the email (Feb 2026 findings):
  - Overall FTR: ~49%  (STP inflates this)
  - Commercial Property non-STP FTR: ~23%
  - Group C (5+ reassignments): ~73% of non-FTR non-STP claims
  - Group A (1-2 extra) is statistically insignificant

Group distribution used for 200 claims (77% non-STP):
  Group 0  FTR            23%  →  46 claims   1 assignment each
  Group A  2-3 events      7%  →  14 claims
  Group B  3-4 events     14%  →  28 claims
  Group C  5-12 events    56%  → 112 claims
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
from pathlib import Path

random.seed(2024)
np.random.seed(2024)

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# ── Reference data ─────────────────────────────────────────────────────────────
LOSS_CAUSES = ["Fire - Electrical", "Fire - Structure", "Wind/Hail - Roof",
               "Wind/Hail - Exterior", "Flood - Sewer Backup", "Flood - Surface",
               "Vandalism", "Equipment Breakdown", "Water Damage - Pipe Burst",
               "Business Interruption"]
POLICY_TYPES = ["BOP", "Commercial Package", "Standalone Property", "Industrial"]
STATES = ["OH", "PA", "VA", "IL", "KY", "NY", "MD", "IN", "WV", "NC"]
CAUSE_TAGS = [
    "System - Auto Route",
    "System - Rule Triggered",
    "Manual - Supervisor",
    "Manual - Workload",
    "User Automated",
    "Fallback Cascade",
    "Named Account Bypass",
]

# Adjusters for Commercial Property
CP_ADJUSTERS = [
    ("ADJ-004", "Marcus Thompson",   "Columbus Property Team 3",    0.71, 31, 45),
    ("ADJ-008", "Robert Harrington", "Pittsburgh Property Team 2",  0.65, 36, 50),
    ("ADJ-014", "Rachel Coleman",    "Senior Adjusters Unit",       0.85, 10, 15),
    ("ADJ-009", "Kevin Walsh",       "CAT Response Team - Central", 0.62, 82, 90),
    ("ADJ-006", "David O'Brien",     "ISS Special Investigations",  0.90, 15, 20),
    ("ADJ-001", "Sarah Mitchell",    "West Zone Ops Ohio Team 6",   0.68, 38, 50),
    ("ADJ-015", "Christopher Adams", "Subrogation Recovery Unit",   0.80, 20, 30),
]
# (adj_id, name, group, historical_ftr_rate, active_claims, max_capacity)

# ── Helper functions ───────────────────────────────────────────────────────────
def rand_date(start="2024-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return s + timedelta(days=random.randint(0, (e - s).days))

def rand_amount(loss_cause):
    base = {
        "Fire - Electrical": (80000, 400000),
        "Fire - Structure":  (150000, 800000),
        "Wind/Hail - Roof":  (15000, 120000),
        "Wind/Hail - Exterior": (10000, 80000),
        "Flood - Sewer Backup": (20000, 150000),
        "Flood - Surface":   (30000, 250000),
        "Vandalism":         (5000, 40000),
        "Equipment Breakdown": (8000, 60000),
        "Water Damage - Pipe Burst": (12000, 90000),
        "Business Interruption": (50000, 500000),
    }
    lo, hi = base.get(loss_cause, (10000, 100000))
    return round(random.uniform(lo, hi), -2)

def group_n_events(group):
    return {
        "0": 1,
        "A": random.randint(2, 3),
        "B": random.randint(3, 4),
        "C": random.randint(5, 12),
    }[group]

def assign_complexity(group, loss_cause):
    if group == "0":
        return random.choices(["Low", "Medium"], weights=[0.6, 0.4])[0]
    if group == "A":
        return random.choices(["Low", "Medium"], weights=[0.3, 0.7])[0]
    if group == "B":
        return random.choices(["Medium", "High"], weights=[0.5, 0.5])[0]
    return random.choices(["Medium", "High"], weights=[0.2, 0.8])[0]

# ── Note templates (Step 6 will classify these) ────────────────────────────────
def make_note(event_type, adj_name, group_name, action):
    if event_type == "Assigned":
        return random.choice([
            f"Claim created and assigned to {adj_name} in group {group_name}. {action}",
            f"FNOL intake complete. Assignment routed to {adj_name} ({group_name}). {action}",
            f"New commercial property claim. Assigned to {adj_name} in {group_name}. {action}",
        ])
    if event_type == "Reassigned":
        return random.choice([
            f"Assigned to {adj_name} in {group_name}. {action}",
            f"Claim transferred to {adj_name} ({group_name}). {action}",
            f"Workload rebalance. Claim moved to {adj_name} in {group_name}. {action}",
            f"Reassigned from prior adjuster to {adj_name} in {group_name}. {action}",
        ])
    if event_type == "Escalated":
        return random.choice([
            f"Claim escalated to {adj_name} in {group_name}. {action}",
            f"Complexity threshold exceeded. Escalated to {adj_name} ({group_name}). {action}",
            f"Large loss trigger. Escalated to senior handler {adj_name} in {group_name}. {action}",
        ])
    return random.choice([
        f"Manually referred to {adj_name} in {group_name}. {action}",
        f"Coverage question. Claim referred to {adj_name} ({group_name}). {action}",
        f"Specialist referral to {adj_name} in {group_name}. {action}",
    ])

ACTIONS = [
    "Action: Structural assessment required; engineer inspection ordered.",
    "Action: Large loss threshold exceeded; authority limit review triggered.",
    "Flag: Activity Overdue — field inspection not completed within SLA.",
    "Action: Business interruption exposure identified; coverage review required.",
    "Action: Subrogation opportunity identified against contractor.",
    "Flag: Adjuster OOO — reassigned to maintain contact SLA.",
    "Action: Contents inventory required; scope of loss expanding.",
    "Action: Building permit pulled; construction timeline extended.",
    "Flag: Coverage dispute — exclusion clause under review.",
    "Action: CAT event declared; CAT team activated for region.",
    "Action: Independent appraisal requested by insured.",
    "Action: Mitigation vendor authorized; drying equipment deployed.",
    "Flag: Litigation hold applied; outside counsel retained.",
    "Action: Supplement estimate submitted; QA review required.",
    "Action: Total loss determination initiated.",
    "Action: Partial payment issued; claim remains open for BI exposure.",
    "Flag: Workload threshold exceeded — system-triggered transfer.",
    "Action: Recorded statement completed; liability analysis underway.",
]

# ── GENERATE: cp_claims_2024.csv ──────────────────────────────────────────────
print("Generating cp_claims_2024.csv ...")
groups = (["0"] * 46 + ["A"] * 14 + ["B"] * 28 + ["C"] * 112)
random.shuffle(groups)

claim_rows = []
for i, grp in enumerate(groups):
    cid = f"CP-{i+1:04d}"
    lc = random.choice(LOSS_CAUSES)
    loss_state = random.choice(STATES)
    insured_state = random.choice(STATES)
    geo_mismatch = int(loss_state != insured_state)
    fnol_date = rand_date()
    days_open = {"0": random.randint(15, 60), "A": random.randint(30, 90),
                 "B": random.randint(60, 150), "C": random.randint(90, 365)}[grp]
    close_date = fnol_date + timedelta(days=days_open)
    stp = random.choices(["Y", "N"], weights=[0.20, 0.80])[0]
    policy = random.choice(POLICY_TYPES)
    amount = rand_amount(lc)
    n_exp = random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
    exp_added_48h = int(n_exp > 1 and random.random() < 0.65)
    complexity = assign_complexity(grp, lc)
    hour = random.randint(6, 20)
    dow = random.randint(0, 6)

    claim_rows.append({
        "claim_id": cid,
        "lob": "Commercial Property",
        "loss_cause": lc,
        "policy_type": policy,
        "loss_state": loss_state,
        "insured_state": insured_state,
        "geographic_mismatch": geo_mismatch,
        "stp_flag": stp,
        "reported_loss_amount": amount,
        "n_exposures_at_fnol": n_exp,
        "exposure_added_within_48h": exp_added_48h,
        "fnol_date": fnol_date.strftime("%m/%d/%Y"),
        "close_date": close_date.strftime("%m/%d/%Y"),
        "fnol_hour": hour,
        "fnol_day_of_week": dow,
        "final_group": grp,
        "complexity": complexity,
    })

df_claims = pd.DataFrame(claim_rows)
df_claims.to_csv(OUT / "cp_claims_2024.csv", index=False)
print(f"  {len(df_claims)} claims | Group dist: {df_claims['final_group'].value_counts().to_dict()}")


# ── GENERATE: cp_assignment_history.csv ───────────────────────────────────────
print("Generating cp_assignment_history.csv ...")
adj_list = [(a[0], a[1], a[2]) for a in CP_ADJUSTERS]

event_type_weights = {"Reassigned": 0.60, "Escalated": 0.25, "Referred": 0.15}

history_rows = []
for _, claim in df_claims.iterrows():
    cid = claim["claim_id"]
    grp = claim["final_group"]
    fnol_dt = datetime.strptime(claim["fnol_date"], "%m/%d/%Y")
    n_events = group_n_events(grp)
    current_time = fnol_dt + timedelta(hours=random.randint(0, 6))

    # pick first adjuster
    adj = random.choice(adj_list)
    adj_id, adj_name, adj_group = adj

    # For Group C, track if same adjuster returns (User-3 pattern)
    seen_adjusters = [adj_id]
    returned = False

    for ev_idx in range(1, n_events + 1):
        if ev_idx == 1:
            ev_type = "Assigned"
            cause = "System - Auto Route"
        else:
            ev_type = random.choices(
                list(event_type_weights.keys()),
                weights=list(event_type_weights.values())
            )[0]
            # cause tag depends on event type
            if ev_type == "Escalated":
                cause = random.choices(
                    ["Manual - Supervisor", "System - Rule Triggered"],
                    weights=[0.70, 0.30]
                )[0]
            elif ev_type == "Referred":
                cause = random.choices(
                    ["Manual - Supervisor", "Named Account Bypass", "User Automated"],
                    weights=[0.60, 0.20, 0.20]
                )[0]
            else:
                cause = random.choices(CAUSE_TAGS[1:], weights=[0.20,0.25,0.20,0.15,0.10,0.10])[0]

            # Occasionally return to a seen adjuster (Group C pattern)
            if grp == "C" and len(seen_adjusters) >= 2 and random.random() < 0.30:
                prev_id = random.choice(seen_adjusters[:-1])
                adj = next((a for a in adj_list if a[0] == prev_id), random.choice(adj_list))
                returned = True
            else:
                adj = random.choice([a for a in adj_list if a[0] != adj_id])

            adj_id, adj_name, adj_group = adj
            if adj_id not in seen_adjusters:
                seen_adjusters.append(adj_id)

        action = random.choice(ACTIONS)
        note = make_note(ev_type, adj_name, adj_group, action)

        # trigger
        if ev_type == "Assigned":
            trigger = random.choices(["System", "Manual"], weights=[0.75, 0.25])[0]
        elif ev_type == "Reassigned":
            trigger = random.choices(["System", "Manual"], weights=[0.45, 0.55])[0]
        else:
            trigger = "Manual"

        history_rows.append({
            "claim_id": cid,
            "assignment_num": ev_idx,
            "event_type": ev_type,
            "cause_tag": cause,
            "timestamp": current_time.strftime("%m/%d/%Y %I:%M %p"),
            "adjuster_id": adj_id,
            "adjuster_name": adj_name,
            "group_name": adj_group,
            "activity_notes": note,
            "trigger": trigger,
            "same_adjuster_returned": int(returned),
            "final_group": grp,
        })

        current_time += timedelta(days=random.randint(0, 7), hours=random.randint(2, 14))

df_history = pd.DataFrame(history_rows)
df_history.to_csv(OUT / "cp_assignment_history.csv", index=False)
print(f"  {len(df_history)} events across {df_claims['claim_id'].nunique()} claims")


# ── GENERATE: cp_adjuster_performance.csv ─────────────────────────────────────
print("Generating cp_adjuster_performance.csv ...")
perf_rows = []
for adj_id, adj_name, adj_group, hist_ftr, active, max_cap in CP_ADJUSTERS:
    n_handled = random.randint(120, 280)
    ftr_n = round(n_handled * hist_ftr)
    avg_days = round(random.uniform(28, 95), 1)
    avg_reassign = round(random.uniform(0.8, 4.2), 2)

    perf_rows.append({
        "adjuster_id": adj_id,
        "adjuster_name": adj_name,
        "group_name": adj_group,
        "cp_claims_handled_2024": n_handled,
        "ftr_count": ftr_n,
        "historical_ftr_rate": round(hist_ftr, 3),
        "avg_cycle_days": avg_days,
        "avg_reassignments_per_claim": avg_reassign,
        "active_open_claims": active,
        "max_capacity": max_cap,
        "utilisation_pct": round(active / max_cap, 3),
        "cp_experience_years": random.randint(2, 18),
        "large_loss_certified": random.choice(["Y", "N", "Y", "Y"]),
    })

df_perf = pd.DataFrame(perf_rows)
df_perf.to_csv(OUT / "cp_adjuster_performance.csv", index=False)
print(f"  {len(df_perf)} adjusters")


# ── GENERATE: cp_claims_2025_pilot.csv (shadow pilot — unseen data) ────────────
print("Generating cp_claims_2025_pilot.csv (shadow mode) ...")
# 40 new 2025 claims — used in Step 10 only
pilot_rows = []
for i in range(40):
    cid = f"CP-2025-{i+1:03d}"
    lc = random.choice(LOSS_CAUSES)
    loss_state = random.choice(STATES)
    insured_state = random.choice(STATES)
    fnol_date = rand_date("2025-01-01", "2025-03-01")
    stp = random.choices(["Y", "N"], weights=[0.20, 0.80])[0]
    policy = random.choice(POLICY_TYPES)
    amount = rand_amount(lc)
    n_exp = random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
    exp_added_48h = int(n_exp > 1 and random.random() < 0.65)
    hour = random.randint(6, 20)
    dow = random.randint(0, 6)
    # Actual outcome: which adjuster ended up as final owner
    actual_final = random.choice(CP_ADJUSTERS)[0]  # adj_id

    pilot_rows.append({
        "claim_id": cid,
        "lob": "Commercial Property",
        "loss_cause": lc,
        "policy_type": policy,
        "loss_state": loss_state,
        "insured_state": insured_state,
        "geographic_mismatch": int(loss_state != insured_state),
        "stp_flag": stp,
        "reported_loss_amount": amount,
        "n_exposures_at_fnol": n_exp,
        "exposure_added_within_48h": exp_added_48h,
        "fnol_date": fnol_date.strftime("%m/%d/%Y"),
        "fnol_hour": hour,
        "fnol_day_of_week": dow,
        "actual_final_adjuster_id": actual_final,
    })

df_pilot = pd.DataFrame(pilot_rows)
df_pilot.to_csv(OUT / "cp_claims_2025_pilot.csv", index=False)
print(f"  {len(df_pilot)} shadow-mode claims")

print("\nAll data files written to pilot/data/")
print("  cp_claims_2024.csv")
print("  cp_assignment_history.csv")
print("  cp_adjuster_performance.csv")
print("  cp_claims_2025_pilot.csv")
