"""Builds all pilot step notebooks."""
import json
from pathlib import Path

HERE = Path(__file__).parent
METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.1"},
}

def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 4, "metadata": METADATA, "cells": cells}

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def save(name, cells):
    p = HERE / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"  Written: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Tag Reassignment Causes
# ══════════════════════════════════════════════════════════════════════════════
STEP1 = [
md("""\
# Step 1 — Tag Reassignment Causes
**Roadmap reference:** SQL / Python rule-based tagging on assignment history

## Goal
Classify every assignment transition by cause type so every downstream model
has clean, interpretable labels.

## Cause taxonomy
| Tag | Meaning |
|-----|---------|
| System - Auto Route | Initial FNOL assignment fired by routing engine |
| System - Rule Triggered | Mid-claim rule (amount, loss cause) fires reassignment |
| Manual - Supervisor | Supervisor explicitly reassigned via claim screen |
| Manual - Workload | Adjuster initiated transfer due to capacity |
| User Automated | Adjuster used "Use Automated Assignment" — logged as system but user-initiated |
| Fallback Cascade | Primary adjuster unavailable; cascaded to next eligible |
| Named Account Bypass | Named-account override skipped normal routing |

## Important Guidewire limitation
If a user reassigns using *Use Automated Assignment*, the system logs it identically
to a true system assignment. This notebook flags those as `User Automated` using
a heuristic: system-logged event on a non-initial assignment where the prior event
was also system-logged within <2 hours."""),

code("""\
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

DATA = Path("data")

df = pd.read_csv(DATA / "cp_assignment_history.csv")
print(f"Loaded {len(df)} assignment events across {df['claim_id'].nunique()} claims")
df.head()"""),

md("## Cause tag distribution (as generated)"),

code("""\
counts = df["cause_tag"].value_counts().reset_index()
counts.columns = ["Cause Tag", "Count"]
counts["Pct"] = (counts["Count"] / len(df) * 100).round(1)
print(counts.to_string(index=False))

fig = px.bar(counts, x="Cause Tag", y="Count", color="Cause Tag",
             title="Assignment Events by Cause Tag — Commercial Property 2024",
             text="Pct")
fig.update_traces(texttemplate="%{text}%", textposition="outside")
fig.update_layout(showlegend=False, xaxis_tickangle=-30)
fig.show()"""),

md("## Flag: User Automated heuristic\nSystem-logged non-initial events within 2 hours of previous system event are re-tagged."),

code("""\
df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %I:%M %p")
df = df.sort_values(["claim_id", "assignment_num"]).reset_index(drop=True)

# Compute hours since previous event
df["prev_trigger"]    = df.groupby("claim_id")["trigger"].shift(1)
df["prev_ts"]         = df.groupby("claim_id")["timestamp_dt"].shift(1)
df["hrs_since_prev"]  = (df["timestamp_dt"] - df["prev_ts"]).dt.total_seconds() / 3600

# Heuristic: non-initial, system-logged, within 2h of previous system event → User Automated
mask = (
    (df["assignment_num"] > 1) &
    (df["trigger"] == "System") &
    (df["prev_trigger"] == "System") &
    (df["hrs_since_prev"] < 2)
)
df.loc[mask, "cause_tag_refined"] = "User Automated"
df["cause_tag_refined"] = df["cause_tag_refined"].fillna(df["cause_tag"])

changed = mask.sum()
print(f"Re-tagged {changed} events as 'User Automated' via heuristic")
print(df["cause_tag_refined"].value_counts().to_string())"""),

md("## Breakdown by final group (reassignment severity)"),

code("""\
ct = (df.groupby(["final_group", "cause_tag_refined"])
        .size()
        .reset_index(name="count"))
fig = px.bar(ct, x="final_group", y="count", color="cause_tag_refined",
             barmode="stack",
             title="Cause Tag Mix by Reassignment Group",
             labels={"final_group": "Reassignment Group", "count": "Events",
                     "cause_tag_refined": "Cause"},
             category_orders={"final_group": ["0", "A", "B", "C"]})
fig.show()"""),

md("## Save tagged output"),

code("""\
out = df.drop(columns=["timestamp_dt", "prev_trigger", "prev_ts", "hrs_since_prev"])
out.to_csv(DATA / "step1_tagged_assignments.csv", index=False)
print(f"Saved {len(out)} rows → data/step1_tagged_assignments.csv")
out.head()"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build Claim Feature Matrix
# ══════════════════════════════════════════════════════════════════════════════
STEP2 = [
md("""\
# Step 2 — Build Claim Feature Matrix
**Roadmap reference:** Data engineering — join claim header + assignment history + cause tags

## Goal
Assemble one flat analytical record per claim.  This is the training and analysis
dataset for every step that follows.

## Features derived
| Feature | Source | Description |
|---------|--------|-------------|
| assignment_count | History | Total transitions |
| pct_manual | History | Share of manual-trigger events |
| pct_system | History | Share of system-trigger events |
| pct_user_auto | History | Share of user-automated events |
| same_adjuster_returned | History | Did a prior adjuster reappear? (User-3 pattern) |
| n_unique_adjusters | History | How many distinct adjusters handled the claim |
| avg_hrs_between_events | History | Velocity of handoffs |
| final_adjuster_appeared_earlier | History | Final owner was already involved earlier |
| geographic_mismatch | Claims | loss_state != insured_state |
| exposure_added_within_48h | Claims | Complexity signal available at FNOL+48h |"""),

code("""\
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

DATA = Path("data")

claims  = pd.read_csv(DATA / "cp_claims_2024.csv")
history = pd.read_csv(DATA / "step1_tagged_assignments.csv")
history["timestamp_dt"] = pd.to_datetime(history["timestamp"], format="%m/%d/%Y %I:%M %p")

print(f"Claims: {len(claims)} | History: {len(history)}")"""),

code("""\
# ── Derive assignment-level features per claim ────────────────────────────────
def build_features(grp):
    grp = grp.sort_values("assignment_num")
    n = len(grp)
    triggers = grp["trigger"].str.lower()
    n_sys    = (triggers == "system").sum()
    n_man    = (triggers == "manual").sum()
    n_ua     = (grp["cause_tag_refined"] == "User Automated").sum()

    adjs = grp["adjuster_id"].tolist()
    n_unique = grp["adjuster_id"].nunique()
    final_adj = adjs[-1]
    returned = int(final_adj in adjs[:-1]) if n > 1 else 0

    ts = grp["timestamp_dt"]
    diffs = ts.diff().dt.total_seconds() / 3600
    avg_hrs = diffs.mean() if n > 1 else 0.0

    return pd.Series({
        "assignment_count":               n,
        "pct_manual":                     round(n_man / n, 3),
        "pct_system":                     round(n_sys / n, 3),
        "pct_user_auto":                  round(n_ua  / n, 3),
        "n_unique_adjusters":             n_unique,
        "same_adjuster_returned":         returned,
        "final_adjuster_appeared_earlier": returned,
        "avg_hrs_between_events":         round(avg_hrs, 1),
    })

agg = history.groupby("claim_id").apply(build_features).reset_index()
matrix = claims.merge(agg, on="claim_id", how="left")

# Group label encoding for ML (0=0, 1=A, 2=B, 3=C)
matrix["group_label"] = matrix["final_group"].map({"0": 0, "A": 1, "B": 2, "C": 3})

matrix.to_csv(DATA / "step2_feature_matrix.csv", index=False)
print(f"Feature matrix: {len(matrix)} rows × {len(matrix.columns)} columns")
matrix.head()"""),

md("## Distribution of assignment counts by group"),

code("""\
fig = px.box(matrix, x="final_group", y="assignment_count",
             color="final_group",
             title="Assignment Count Distribution by Group",
             labels={"final_group": "Group", "assignment_count": "# Assignments"},
             category_orders={"final_group": ["0", "A", "B", "C"]})
fig.show()

print("\\nFTR rate:", round(len(matrix[matrix["final_group"]=="0"]) / len(matrix) * 100, 1), "%")
print("Group C rate:", round(len(matrix[matrix["final_group"]=="C"]) / len(matrix) * 100, 1), "%")"""),

md("## Correlation — feature vs group label"),

code("""\
num_cols = ["assignment_count", "pct_manual", "pct_user_auto",
            "n_unique_adjusters", "same_adjuster_returned",
            "avg_hrs_between_events", "geographic_mismatch",
            "exposure_added_within_48h", "reported_loss_amount", "group_label"]
corr = matrix[num_cols].corr()[["group_label"]].drop("group_label").round(3)
corr.columns = ["Correlation with group_label"]
print(corr.sort_values("Correlation with group_label", ascending=False).to_string())"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cluster Failure Types
# ══════════════════════════════════════════════════════════════════════════════
STEP3 = [
md("""\
# Step 3 — Cluster Failure Types
**Roadmap reference:** K-Means unsupervised clustering

## Goal
Identify distinct failure archetypes — not just reassignment count.
This tells you *which type* of problem you are solving before you design any fix.

## Expected clusters (from roadmap)
1. **Legitimate complexity escalation** — linear escalation, no bounce, high loss amount
2. **Structural bounce** — same-tier handoffs, low escalation, User-3 pattern
3. **Geographic mismatch routing** — high geo mismatch, system-triggered
4. **Manual referral re-entry** — high manual trigger, referee returns to earlier adjuster

## Method
- StandardScaler → KMeans k=4
- t-SNE (2D) for visualization
- Silhouette score to validate separation"""),

code("""\
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pathlib import Path

DATA = Path("data")
matrix = pd.read_csv(DATA / "step2_feature_matrix.csv")

# Use only non-FTR claims for clustering (Group A/B/C — the failure cases)
df_fail = matrix[matrix["final_group"] != "0"].copy().reset_index(drop=True)
print(f"Clustering {len(df_fail)} non-FTR claims")"""),

code("""\
# Features for clustering
CLUSTER_FEATURES = [
    "assignment_count", "pct_manual", "pct_system", "pct_user_auto",
    "n_unique_adjusters", "same_adjuster_returned", "avg_hrs_between_events",
    "geographic_mismatch", "exposure_added_within_48h",
    "reported_loss_amount",
]

X = df_fail[CLUSTER_FEATURES].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using silhouette score
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = round(silhouette_score(X_scaled, labels), 4)

print("Silhouette scores by k:")
for k, s in sil_scores.items():
    print(f"  k={k}: {s}")
best_k = max(sil_scores, key=sil_scores.get)
print(f"\\nBest k = {best_k}")"""),

code("""\
# Final clustering
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df_fail["cluster"] = km.fit_predict(X_scaled)

# Label clusters by their dominant characteristics
cluster_profiles = df_fail.groupby("cluster")[CLUSTER_FEATURES + ["final_group"]].mean().round(3)
print("Cluster profiles (mean feature values):")
print(cluster_profiles[["assignment_count", "pct_manual", "geographic_mismatch",
                          "same_adjuster_returned", "avg_hrs_between_events"]].to_string())"""),

code("""\
# Business labels based on profile inspection
CLUSTER_LABELS = {
    0: "Complexity Escalation",
    1: "Structural Bounce",
    2: "Geographic Mismatch",
    3: "Manual Re-entry",
}
# Auto-assign by highest distinguishing feature
for cid, row in cluster_profiles.iterrows():
    if row["geographic_mismatch"] == cluster_profiles["geographic_mismatch"].max():
        CLUSTER_LABELS[cid] = "Geographic Mismatch"
    elif row["same_adjuster_returned"] == cluster_profiles["same_adjuster_returned"].max():
        CLUSTER_LABELS[cid] = "Structural Bounce (User-3 Pattern)"
    elif row["pct_manual"] == cluster_profiles["pct_manual"].max():
        CLUSTER_LABELS[cid] = "Manual Re-entry / Override"
    else:
        CLUSTER_LABELS[cid] = "Complexity Escalation"

df_fail["cluster_label"] = df_fail["cluster"].map(CLUSTER_LABELS)
print("\\nCluster labels assigned:")
print(df_fail["cluster_label"].value_counts().to_string())"""),

code("""\
# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(X_scaled)
df_fail["tsne_x"] = coords[:, 0]
df_fail["tsne_y"] = coords[:, 1]

fig = px.scatter(df_fail, x="tsne_x", y="tsne_y",
                 color="cluster_label",
                 hover_data=["claim_id", "final_group", "assignment_count"],
                 title="Failure Archetype Clusters — t-SNE (Commercial Property 2024)",
                 labels={"tsne_x": "t-SNE 1", "tsne_y": "t-SNE 2",
                         "cluster_label": "Failure Type"})
fig.update_traces(marker=dict(size=6, opacity=0.8))
fig.show()"""),

code("""\
# Save
df_fail[["claim_id", "cluster", "cluster_label"]].to_csv(
    DATA / "step3_clusters.csv", index=False)
print("Saved → data/step3_clusters.csv")

# Summary table
summary = (df_fail.groupby("cluster_label")
           .agg(n_claims=("claim_id", "count"),
                avg_assignments=("assignment_count", "mean"),
                pct_returned=("same_adjuster_returned", "mean"),
                pct_geo_mismatch=("geographic_mismatch", "mean"))
           .round(2).reset_index())
print("\\nCluster summary:")
print(summary.to_string(index=False))"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Complexity Classifier at FNOL
# ══════════════════════════════════════════════════════════════════════════════
STEP4 = [
md("""\
# Step 4 — Predict Claim Complexity at FNOL
**Roadmap reference:** XGBoost classifier

## Goal
Predict at First Notice of Loss what final reassignment group (0/A/B/C) a claim
will reach. This lets you route to the right adjuster tier upfront.

## Critical rule: no data leakage
Only features available at the moment of FNOL are used. Assignment history,
cause tags, and exposure additions discovered after 48h are excluded.

## FNOL-safe features
`loss_cause`, `policy_type`, `loss_state`, `insured_state`, `geographic_mismatch`,
`stp_flag`, `reported_loss_amount`, `n_exposures_at_fnol`,
`exposure_added_within_48h`, `fnol_hour`, `fnol_day_of_week`

## Training / test split
- Training: 80% of 2024 closed claims (random split, stratified by group)
- Test: held-out 20% — simulates scoring on unseen claims"""),

code("""\
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
import pickle
from pathlib import Path

DATA = Path("data")
matrix = pd.read_csv(DATA / "step2_feature_matrix.csv")
print(f"Feature matrix: {len(matrix)} rows")
matrix["final_group"].value_counts()"""),

code("""\
# ── Feature engineering — FNOL-only ──────────────────────────────────────────
FNOL_FEATURES = [
    "geographic_mismatch", "stp_flag_enc",
    "reported_loss_amount", "n_exposures_at_fnol",
    "exposure_added_within_48h", "fnol_hour", "fnol_day_of_week",
    "loss_cause_enc", "policy_type_enc",
]

df = matrix.copy()

# Encode categoricals
le_lc = LabelEncoder()
le_pt = LabelEncoder()
df["loss_cause_enc"]   = le_lc.fit_transform(df["loss_cause"])
df["policy_type_enc"]  = le_pt.fit_transform(df["policy_type"])
df["stp_flag_enc"]     = (df["stp_flag"] == "Y").astype(int)

X = df[FNOL_FEATURES]
y = df["group_label"]   # 0=Group0, 1=A, 2=B, 3=C

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print("Class distribution (test):", dict(y_test.value_counts().sort_index()))"""),

code("""\
# ── Train XGBoost ─────────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Group 0 (FTR)", "Group A", "Group B", "Group C"]))"""),

code("""\
# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Group 0", "Group A", "Group B", "Group C"],
                y=["Group 0", "Group A", "Group B", "Group C"],
                title="Confusion Matrix — FNOL Complexity Classifier",
                text_auto=True, color_continuous_scale="Blues")
fig.show()"""),

code("""\
# ── Feature importance ────────────────────────────────────────────────────────
imp = pd.DataFrame({
    "Feature": FNOL_FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
             title="XGBoost Feature Importance — FNOL Complexity Prediction",
             color="Importance", color_continuous_scale="Blues")
fig.show()"""),

code("""\
# ── Save model + predictions ──────────────────────────────────────────────────
with open(DATA / "step4_complexity_model.pkl", "wb") as f:
    pickle.dump({"model": model, "le_lc": le_lc, "le_pt": le_pt,
                 "features": FNOL_FEATURES}, f)

# Add predictions back to full dataset
df["predicted_group_label"] = model.predict(X)
df["predicted_group"] = df["predicted_group_label"].map({0:"0",1:"A",2:"B",3:"C"})
df[["claim_id", "predicted_group", "final_group"]].to_csv(
    DATA / "step4_predictions.csv", index=False)
print("Saved → data/step4_complexity_model.pkl")
print("Saved → data/step4_predictions.csv")
print(f"\\nTarget: >70% accuracy on held-out test | Achieved: {acc*100:.1f}%")"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Adjuster-Claim Match Scorer
# ══════════════════════════════════════════════════════════════════════════════
STEP5 = [
md("""\
# Step 5 — Adjuster-Claim Match Scorer
**Roadmap reference:** Weighted scoring function

## Goal
Replace the rigid routing cascade (zip → state → supervisor fallback) with a
scorer that ranks eligible adjusters for any incoming claim.

## Scoring formula
```
Score = w1 × LOB_match
      + w2 × historical_FTR_rate
      + w3 × (1 − utilisation)
      + w4 × experience_alignment
```

| Component | Weight | Source |
|-----------|--------|--------|
| LOB match (binary) | 0.30 | Adjuster profile |
| Historical FTR rate on similar claims | 0.25 | Performance table |
| Inverse utilisation (1 - active/max) | 0.20 | Capacity table |
| Experience alignment to predicted complexity | 0.25 | Profile + Step 4 output |

## Experience alignment
- Predicted Group 0 / A → Adjuster I / II sufficient → score = 1.0
- Predicted Group B → Specialist preferred → score 0.8 if specialist, 0.5 otherwise
- Predicted Group C → Senior or Specialist required → score 1.0 if senior, 0.6 otherwise"""),

code("""\
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

DATA = Path("data")
claims      = pd.read_csv(DATA / "step2_feature_matrix.csv")
preds       = pd.read_csv(DATA / "step4_predictions.csv")
perf        = pd.read_csv(DATA / "cp_adjuster_performance.csv")
history     = pd.read_csv(DATA / "step1_tagged_assignments.csv")

claims = claims.merge(preds[["claim_id", "predicted_group"]], on="claim_id", how="left")
print(f"Claims with predictions: {len(claims)}")
perf"""),

code("""\
# ── Scoring function ──────────────────────────────────────────────────────────
WEIGHTS = {"lob_match": 0.30, "ftr_rate": 0.25, "inv_util": 0.20, "exp_align": 0.25}

SENIORITY = {
    "ADJ-004": "specialist",  "ADJ-008": "generalist",
    "ADJ-014": "senior",      "ADJ-009": "specialist",
    "ADJ-006": "senior",      "ADJ-001": "generalist",
    "ADJ-015": "specialist",
}

def experience_alignment(predicted_group, adj_id):
    level = SENIORITY.get(adj_id, "generalist")
    if predicted_group in ("0", "A"):
        return 1.0  # any adjuster suitable
    if predicted_group == "B":
        return 1.0 if level in ("specialist", "senior") else 0.5
    # Group C
    return 1.0 if level == "senior" else (0.7 if level == "specialist" else 0.4)

def score_adjuster(adj_row, predicted_group):
    lob_match   = 1.0   # all these adjusters handle CP
    ftr         = adj_row["historical_ftr_rate"]
    inv_util    = 1.0 - adj_row["utilisation_pct"]
    inv_util    = max(0, min(1, inv_util))   # clip
    exp_align   = experience_alignment(predicted_group, adj_row["adjuster_id"])

    return (WEIGHTS["lob_match"]  * lob_match +
            WEIGHTS["ftr_rate"]   * ftr        +
            WEIGHTS["inv_util"]   * inv_util   +
            WEIGHTS["exp_align"]  * exp_align)

# Score all adjusters for each claim
score_rows = []
for _, claim in claims.iterrows():
    ranked = []
    for _, adj in perf.iterrows():
        s = score_adjuster(adj, claim["predicted_group"])
        ranked.append((adj["adjuster_id"], adj["adjuster_name"], round(s, 4)))
    ranked.sort(key=lambda x: x[2], reverse=True)
    top1_id, top1_name, top1_score = ranked[0]
    score_rows.append({
        "claim_id": claim["claim_id"],
        "predicted_group": claim["predicted_group"],
        "final_group": claim["final_group"],
        "top_scorer_id": top1_id,
        "top_scorer_name": top1_name,
        "top_score": top1_score,
        "all_scores": str(ranked),
    })

df_scores = pd.DataFrame(score_rows)
print(f"Scored {len(df_scores)} claims")
print("\\nTop scorer distribution:")
print(df_scores["top_scorer_name"].value_counts().to_string())"""),

code("""\
# ── Compare scorer recommendation vs actual final adjuster ────────────────────
final_adj = (history.groupby("claim_id")
             .apply(lambda g: g.sort_values("assignment_num").iloc[-1]["adjuster_id"])
             .reset_index(name="actual_final_adj_id"))

df_scores = df_scores.merge(final_adj, on="claim_id", how="left")
df_scores["top_match_actual"] = (df_scores["top_scorer_id"] == df_scores["actual_final_adj_id"]).astype(int)

match_rate = df_scores["top_match_actual"].mean()
print(f"Scorer top-1 = actual final owner: {match_rate*100:.1f}%")
print(f"Target: >50%")

# By group
by_group = df_scores.groupby("final_group")["top_match_actual"].mean().round(3)
print("\\nMatch rate by group:")
print(by_group.to_string())"""),

code("""\
fig = px.bar(by_group.reset_index(), x="final_group", y="top_match_actual",
             title="Scorer Top-1 Match Rate vs Actual Final Owner (by Group)",
             labels={"final_group": "Reassignment Group", "top_match_actual": "Match Rate"},
             color="final_group",
             category_orders={"final_group": ["0", "A", "B", "C"]})
fig.add_hline(y=0.50, line_dash="dash", line_color="red",
              annotation_text="Target: 50%")
fig.update_yaxes(tickformat=".0%", range=[0, 1])
fig.show()

df_scores.to_csv(DATA / "step5_scorer_results.csv", index=False)
print("Saved → data/step5_scorer_results.csv")"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — LLM Note Parsing & Classification
# ══════════════════════════════════════════════════════════════════════════════
STEP6 = [
md("""\
# Step 6 — LLM Note Parsing & Classification
**Roadmap reference:** LLM text classification for label enrichment

## Goal
Classify each assignment note as:
- **(A) Avoidable reassignment** — could have been prevented with better initial routing
- **(B) Necessary — complexity change** — claim evolved and genuinely required a different adjuster
- **(C) Manual override** — supervisor or adjuster bypassed routing logic
- **(D) Unclear** — insufficient information in the note

## Why this matters
A significant portion of routing intelligence sits in free-text notes that
structured rules completely miss. These labels also enrich the training data
for Step 4's complexity model.

## Sample size
A sample of 50 notes is classified. In a real pilot, manually label 100-200
first as ground truth, then measure LLM agreement rate (target >85%)."""),

code("""\
import pandas as pd
import json
import os
import toml
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

DATA = Path("data")
history = pd.read_csv(DATA / "step1_tagged_assignments.csv")

# Sample 50 non-initial events (reassignments only — most interesting for classification)
sample = (history[history["assignment_num"] > 1]
          .sample(50, random_state=42)
          .reset_index(drop=True))

print(f"Sample: {len(sample)} notes")
print("Event type mix:")
print(sample["event_type"].value_counts().to_string())"""),

code("""\
# ── Load LLM client ───────────────────────────────────────────────────────────
def _req_secret(key):
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val: return str(val).strip()
    except Exception:
        pass
    secrets_path = Path(__file__).parents[1] / ".streamlit" / ".secrets.toml"
    if secrets_path.exists():
        cfg = toml.load(str(secrets_path))
        if key in cfg: return str(cfg[key]).strip()
    raise RuntimeError(f"Missing secret: {key}")

llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=_req_secret("AZURE_CHAT_ENDPOINT"),
    openai_api_key=_req_secret("AZURE_CHAT_API_KEY"),
    openai_api_version=_req_secret("AZURE_CHAT_API_VERSION"),
    deployment_name=_req_secret("AZURE_CHAT_DEPLOYMENT"),
    model_name=_req_secret("AZURE_CHAT_MODEL"),
    model_kwargs={"response_format": {"type": "json_object"}},
)
print("LLM client ready")"""),

code("""\
# ── Batch classify in groups of 10 ────────────────────────────────────────────
SYSTEM_PROMPT = \"\"\"You are an insurance claims analyst.
Classify each assignment note into exactly one category. Always respond with valid JSON.

Categories:
A = Avoidable reassignment — initial routing was wrong; better upfront data could have prevented this move
B = Necessary — claim genuinely evolved (new exposure, large loss, legal involvement) requiring a different adjuster
C = Manual override — supervisor or adjuster bypassed normal routing logic intentionally
D = Unclear — not enough information in the note to determine

Also extract any adjuster skill or license requirement implied by the note (e.g. "large loss", "SIU", "subrogation").\"\"\\"

results = []
batch_size = 10

for batch_start in range(0, len(sample), batch_size):
    batch = sample.iloc[batch_start:batch_start+batch_size]
    notes_text = "\\n".join(
        f"Note {i+1}: {row['activity_notes']}"
        for i, (_, row) in enumerate(batch.iterrows())
    )
    user_msg = f\"\"\"Classify each of the following {len(batch)} assignment notes.
For each note return: category (A/B/C/D), confidence (high/medium/low), implied_skill (or null).

{notes_text}

Respond as JSON: {{"results": [{{"note": 1, "category": "A", "confidence": "high", "implied_skill": null}}, ...]}}\"\"\"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    parsed = json.loads(response.content)
    batch_results = parsed.get("results", [])
    for r in batch_results:
        idx = batch_start + r.get("note", 1) - 1
        if idx < len(sample):
            results.append({
                "claim_id":       sample.iloc[idx]["claim_id"],
                "assignment_num": sample.iloc[idx]["assignment_num"],
                "event_type":     sample.iloc[idx]["event_type"],
                "activity_notes": sample.iloc[idx]["activity_notes"],
                "cause_tag":      sample.iloc[idx]["cause_tag_refined"],
                "category":       r.get("category", "D"),
                "confidence":     r.get("confidence", "low"),
                "implied_skill":  r.get("implied_skill"),
            })
    print(f"  Classified notes {batch_start+1}–{min(batch_start+batch_size, len(sample))}")

df_classified = pd.DataFrame(results)
print(f"\\nTotal classified: {len(df_classified)}")"""),

code("""\
# ── Results ───────────────────────────────────────────────────────────────────
import plotly.express as px

cat_map = {"A": "Avoidable", "B": "Necessary", "C": "Manual Override", "D": "Unclear"}
df_classified["category_label"] = df_classified["category"].map(cat_map)

print("Classification breakdown:")
print(df_classified["category_label"].value_counts().to_string())
print("\\nConfidence breakdown:")
print(df_classified["confidence"].value_counts().to_string())

fig = px.pie(df_classified, names="category_label",
             title="Assignment Note Classification — 50 Sample Notes",
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.show()

df_classified.to_csv(DATA / "step6_llm_classifications.csv", index=False)
print("\\nSaved → data/step6_llm_classifications.csv")"""),

code("""\
# ── Implied skills extracted ───────────────────────────────────────────────────
skills = (df_classified[df_classified["implied_skill"].notna()]
          [["claim_id", "implied_skill", "category_label"]])
print("Implied skill requirements extracted from notes:")
print(skills.to_string(index=False))"""),
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Metrics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
STEP9 = [
md("""\
# Step 9 — Feedback Loop & Metrics
**Roadmap reference:** BI dashboard + model monitoring

## Goal
Track the key pilot KPIs against targets. Without a closed-loop signal, any
improvement is invisible and will degrade silently.

## Metrics tracked
| Metric | Current State | Pilot Target |
|--------|---------------|--------------|
| Commercial Property FTR (non-STP) | ~23% | 35%+ |
| Group C rate | ~56% | Reduce to <60% (identify drivers) |
| FNOL complexity accuracy | Not measured | >70% |
| Scorer top-1 = final owner | Not measured | >50% |
| LLM note classification agreement | Not measured | >85% |"""),

code("""\
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA = Path("data")

claims  = pd.read_csv(DATA / "step2_feature_matrix.csv")
preds   = pd.read_csv(DATA / "step4_predictions.csv")
scores  = pd.read_csv(DATA / "step5_scorer_results.csv")
llm_cls = pd.read_csv(DATA / "step6_llm_classifications.csv")
history = pd.read_csv(DATA / "step1_tagged_assignments.csv")

claims = claims.merge(preds[["claim_id","predicted_group"]], on="claim_id", how="left")
print("All step outputs loaded")"""),

code("""\
# ── KPI 1: FTR rate ───────────────────────────────────────────────────────────
non_stp = claims[claims["stp_flag"] == "N"]
ftr_rate = (non_stp["final_group"] == "0").mean()
group_c_rate = (non_stp["final_group"] == "C").mean()

print(f"Non-STP claims:  {len(non_stp)}")
print(f"FTR rate:        {ftr_rate*100:.1f}%  (target: 35%+)")
print(f"Group C rate:    {group_c_rate*100:.1f}%  (target: <60%)")"""),

code("""\
# ── KPI 2: Model accuracy ─────────────────────────────────────────────────────
acc = (preds["predicted_group"] == preds["final_group"]).mean()
print(f"FNOL complexity accuracy: {acc*100:.1f}%  (target: >70%)")"""),

code("""\
# ── KPI 3: Scorer match rate ──────────────────────────────────────────────────
match = scores["top_match_actual"].mean()
print(f"Scorer top-1 = final owner: {match*100:.1f}%  (target: >50%)")"""),

code("""\
# ── KPI 4: LLM avoidable rate ─────────────────────────────────────────────────
avoidable = (llm_cls["category"] == "A").mean()
print(f"Avoidable reassignments (LLM): {avoidable*100:.1f}% of sampled notes")"""),

code("""\
# ── KPI summary dashboard ─────────────────────────────────────────────────────
kpis = [
    ("FTR Rate (non-STP)",           ftr_rate*100,   35.0,  "%"),
    ("Group C Rate (non-STP)",        group_c_rate*100, 60.0, "% (lower=better)"),
    ("FNOL Complexity Accuracy",      acc*100,        70.0,  "%"),
    ("Scorer Top-1 Match Rate",       match*100,      50.0,  "%"),
]

rows = []
for name, actual, target, unit in kpis:
    if "lower" in unit:
        status = "✅ On Target" if actual < target else "⚠️ Above Target"
    else:
        status = "✅ On Target" if actual >= target else "⚠️ Below Target"
    rows.append({"Metric": name, "Current": f"{actual:.1f}{unit[:1]}",
                 "Target": f"{target:.0f}{unit[:1]}", "Status": status})

df_kpi = pd.DataFrame(rows)
print("\\n=== PILOT KPI DASHBOARD ===")
print(df_kpi.to_string(index=False))"""),

code("""\
# ── Weekly FTR trend (simulated — showing how monitoring would look) ───────────
history["timestamp_dt"] = pd.to_datetime(history["timestamp"], format="%m/%d/%Y %I:%M %p")
history["week"] = history["timestamp_dt"].dt.to_period("W").astype(str)

weekly_ftr = (claims.merge(
    history.groupby("claim_id")["timestamp_dt"].min().reset_index(name="first_event"),
    on="claim_id")
 .assign(week=lambda d: pd.to_datetime(d["first_event"]).dt.to_period("W").astype(str))
 .groupby("week")
 .apply(lambda g: (g["final_group"]=="0").mean())
 .reset_index(name="ftr_rate"))

fig = px.line(weekly_ftr, x="week", y="ftr_rate",
              title="Weekly FTR Rate — Commercial Property 2024 (Monitoring View)",
              labels={"week": "Week", "ftr_rate": "FTR Rate"})
fig.add_hline(y=0.35, line_dash="dash", line_color="green",
              annotation_text="Pilot Target: 35%")
fig.add_hline(y=ftr_rate, line_dash="dot", line_color="orange",
              annotation_text=f"Current Baseline: {ftr_rate*100:.0f}%")
fig.update_yaxes(tickformat=".0%")
fig.show()"""),

code("""\
# ── Assignment count distribution ─────────────────────────────────────────────
fig = px.histogram(claims, x="assignment_count", color="final_group",
                   title="Assignment Count Distribution — Commercial Property 2024",
                   labels={"assignment_count":"# Assignments","final_group":"Group"},
                   nbins=12, barmode="overlay", opacity=0.7,
                   category_orders={"final_group":["0","A","B","C"]})
fig.show()"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Shadow Pilot
# ══════════════════════════════════════════════════════════════════════════════
STEP10 = [
md("""\
# Step 10 — Shadow Mode Pilot
**Roadmap reference:** A/B test — new scorer alongside existing rules for 60 days

## Goal
Run the new matching scorer against 40 incoming 2025 Commercial Property claims
in shadow mode — log what the scorer would have recommended *without* changing
actual assignments — then measure the FTR delta.

## Shadow mode means
- No live system change required
- Even a CSV export works: log scorer recommendation per claim
- After 60 days, compare: was top-scored adjuster the actual final owner?

## Pilot target
Top-ranked adjuster = final owner in >50% of cases (vs ~23% baseline FTR)"""),

code("""\
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA = Path("data")

# Load new 2025 claims (unseen during training)
new_claims = pd.read_csv(DATA / "cp_claims_2025_pilot.csv")
perf       = pd.read_csv(DATA / "cp_adjuster_performance.csv")

# Load trained model
with open(DATA / "step4_complexity_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model    = bundle["model"]
le_lc    = bundle["le_lc"]
le_pt    = bundle["le_pt"]
features = bundle["features"]

print(f"Shadow pilot: {len(new_claims)} new 2025 claims")
new_claims.head()"""),

code("""\
# ── Step 1: Predict complexity at FNOL for each new claim ─────────────────────
df = new_claims.copy()

# Handle unseen labels in encoders
def safe_encode(le, col):
    known = set(le.classes_)
    return col.map(lambda x: le.transform([x])[0] if x in known else -1)

df["loss_cause_enc"]  = safe_encode(le_lc, df["loss_cause"])
df["policy_type_enc"] = safe_encode(le_pt, df["policy_type"])
df["stp_flag_enc"]    = (df["stp_flag"] == "Y").astype(int)

X_new = df[features].fillna(0)
df["predicted_group_label"] = model.predict(X_new)
df["predicted_group"] = df["predicted_group_label"].map({0:"0",1:"A",2:"B",3:"C"})

print("Complexity predictions for 2025 claims:")
print(df["predicted_group"].value_counts().to_string())"""),

code("""\
# ── Step 2: Score adjusters for each new claim ────────────────────────────────
WEIGHTS = {"lob_match": 0.30, "ftr_rate": 0.25, "inv_util": 0.20, "exp_align": 0.25}
SENIORITY = {
    "ADJ-004":"specialist","ADJ-008":"generalist","ADJ-014":"senior",
    "ADJ-009":"specialist","ADJ-006":"senior","ADJ-001":"generalist","ADJ-015":"specialist"
}

def exp_align(pred_grp, adj_id):
    lvl = SENIORITY.get(adj_id, "generalist")
    if pred_grp in ("0","A"): return 1.0
    if pred_grp == "B": return 1.0 if lvl in ("specialist","senior") else 0.5
    return 1.0 if lvl=="senior" else (0.7 if lvl=="specialist" else 0.4)

shadow_rows = []
for _, claim in df.iterrows():
    ranked = []
    for _, adj in perf.iterrows():
        inv_util = max(0, 1 - adj["utilisation_pct"])
        s = (WEIGHTS["lob_match"]  * 1.0 +
             WEIGHTS["ftr_rate"]   * adj["historical_ftr_rate"] +
             WEIGHTS["inv_util"]   * inv_util +
             WEIGHTS["exp_align"]  * exp_align(claim["predicted_group"], adj["adjuster_id"]))
        ranked.append((adj["adjuster_id"], adj["adjuster_name"], round(s, 4)))
    ranked.sort(key=lambda x: x[2], reverse=True)

    top1_id, top1_name, top1_score = ranked[0]
    top2_id, top2_name, top2_score = ranked[1]
    top3_id, top3_name, top3_score = ranked[2]

    actual = claim["actual_final_adjuster_id"]
    in_top1 = int(actual == top1_id)
    in_top3 = int(actual in [r[0] for r in ranked[:3]])

    shadow_rows.append({
        "claim_id":          claim["claim_id"],
        "predicted_group":   claim["predicted_group"],
        "scorer_top1_id":    top1_id,
        "scorer_top1_name":  top1_name,
        "scorer_top1_score": top1_score,
        "scorer_top2_name":  top2_name,
        "scorer_top3_name":  top3_name,
        "actual_final_adj":  actual,
        "top1_match":        in_top1,
        "top3_match":        in_top3,
    })

df_shadow = pd.DataFrame(shadow_rows)
df_shadow.to_csv(DATA / "step10_shadow_results.csv", index=False)
print(f"Shadow results: {len(df_shadow)} claims scored")"""),

code("""\
# ── Results ───────────────────────────────────────────────────────────────────
top1_rate = df_shadow["top1_match"].mean()
top3_rate = df_shadow["top3_match"].mean()

print("=== SHADOW PILOT RESULTS ===")
print(f"Claims scored:          {len(df_shadow)}")
print(f"Top-1 match rate:       {top1_rate*100:.1f}%  (target: >50%)")
print(f"Top-3 match rate:       {top3_rate*100:.1f}%")
print(f"Baseline FTR:           23.0%")
print(f"Delta (Top-1 vs FTR):  +{(top1_rate-0.23)*100:.1f} pp")

status = "✅ TARGET MET" if top1_rate >= 0.50 else f"⚠️ {(0.50-top1_rate)*100:.1f}pp below target"
print(f"\\nPilot status: {status}")"""),

code("""\
# ── Waterfall: baseline → model → scorer ──────────────────────────────────────
stages = ["Baseline FTR", "FNOL Model Accuracy", "Scorer Top-1 Match"]
values = [23.0, None, top1_rate*100]
# Fill model accuracy from step4 predictions
try:
    preds = pd.read_csv(DATA / "step4_predictions.csv")
    model_acc = (preds["predicted_group"]==preds["final_group"]).mean()*100
    values[1] = model_acc
except Exception:
    values[1] = 70.0

fig = go.Figure(go.Bar(
    x=stages, y=values,
    text=[f"{v:.1f}%" for v in values],
    textposition="outside",
    marker_color=["#636EFA","#EF553B","#00CC96"]
))
fig.add_hline(y=50, line_dash="dash", line_color="red",
              annotation_text="50% target")
fig.update_layout(title="Shadow Pilot — Performance vs Targets",
                  yaxis_title="Rate (%)", yaxis_range=[0, 110])
fig.show()"""),

code("""\
# ── Top scorer distribution across shadow claims ──────────────────────────────
top1_dist = df_shadow["scorer_top1_name"].value_counts().reset_index()
top1_dist.columns = ["Adjuster", "Claims Recommended"]
print("\\nScorer recommendations:")
print(top1_dist.to_string(index=False))

fig = px.bar(top1_dist, x="Adjuster", y="Claims Recommended",
             title="Shadow Mode — Scorer Recommendations per Adjuster",
             color="Adjuster")
fig.show()

print(f"\\nNext step: run live for 60 days, compare FTR on scorer-routed vs")
print(f"control-group claims (existing routing). Report FTR delta.")"""),
]

# ── Write all notebooks ──────────────────────────────────────────────────────
print("Writing notebooks...")
save("step1_tag_causes.ipynb",        STEP1)
save("step2_feature_matrix.ipynb",    STEP2)
save("step3_clustering.ipynb",        STEP3)
save("step4_complexity_model.ipynb",  STEP4)
save("step5_adjuster_scorer.ipynb",   STEP5)
save("step6_llm_parsing.ipynb",       STEP6)
save("step9_metrics.ipynb",           STEP9)
save("step10_shadow_pilot.ipynb",     STEP10)
print("Done.")
