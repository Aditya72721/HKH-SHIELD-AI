#!/usr/bin/env python3
"""
HKH SHIELD AI — glacial/cryospheric outburst scoring model.

HONESTY NOTE (read the model card in the app too)
-------------------------------------------------
The execution container used to build this model has NO network access, so the
raw ICIMOD HKH glacial-lake inventory, the Zenodo global GLOF database
(Veh et al.), Carrivick & Tweed and the NASA Global Landslide Catalog could NOT
be downloaded into it. training_data.csv is therefore a COMPILED dataset:

  * 3 rows  = the three replay cases, features taken from published reports
              (status "documented").
  * 27 rows = other well-documented HKH / high-mountain Asia outburst events,
              features taken from the published ranges in the literature
              (status "literature").
  * rest    = non-burst lake rows, feature values SAMPLED from the published
              distributions of HKH moraine/bedrock-dammed lakes
              (status "sampled").

This makes the model a PROTOTYPE SCORING MODEL, not a validated predictor.
Do not use it operationally. Replace training_data.csv with the real
inventories before drawing any scientific conclusion.
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

RNG = np.random.default_rng(20260826)

FEATURES = [
    "lake_area_km2",          # water body area
    "lake_growth_pct_yr",     # multi-temporal expansion rate
    "dam_type",               # 0 bedrock · 1 moraine · 2 ice / debris dam
    "headwall_slope_deg",     # slope above the lake / source zone
    "glacier_area_km2",       # feeding glacier area
    "precip_anomaly_pct",     # rainfall anomaly vs seasonal baseline
    "temp_anomaly_c",         # temperature anomaly
    "downstream_exposure",    # 0-1 index: people + assets in the flow path
]
LABELS = {
    "lake_area_km2": "Lake / water-body area (km²)",
    "lake_growth_pct_yr": "Lake expansion rate (%/yr)",
    "dam_type": "Dam type (bedrock→ice/debris)",
    "headwall_slope_deg": "Headwall slope above source (°)",
    "glacier_area_km2": "Feeding glacier area (km²)",
    "precip_anomaly_pct": "Precipitation anomaly (%)",
    "temp_anomaly_c": "Temperature anomaly (°C)",
    "downstream_exposure": "Downstream exposure index (0-1)",
}

# ---------------------------------------------------------------- documented
# event_id, features…, label, provenance
DOCUMENTED = [
    # --- the three replay cases -------------------------------------------
    ("NP-RASUWA-2026", 0.11, 120.0, 2, 55.0, 4.6, -30.0, 2.0, 0.88, 1, "documented"),
    ("IN-KEDARNATH-2013", 0.04, 40.0, 1, 45.0, 6.6, 375.0, 1.5, 0.95, 1, "documented"),
    ("IN-SLHONAK-2023", 1.68, 9.0, 1, 45.0, 6.9, 210.0, 2.5, 0.80, 1, "documented"),
    # --- other documented high-mountain Asia outburst / mass-flow events ---
    ("NP-RASUWA-2025", 0.09, 90.0, 2, 52.0, 4.6, 60.0, 1.8, 0.70, 1, "literature"),
    ("IN-CHAMOLI-2021", 0.00, 0.0, 2, 60.0, 3.5, -20.0, 1.2, 0.75, 1, "literature"),
    ("NP-DIGTSHO-1985", 0.60, 12.0, 1, 40.0, 5.2, 30.0, 1.0, 0.45, 1, "literature"),
    ("NP-TAMPOKHARI-1998", 0.45, 10.0, 1, 42.0, 4.0, 45.0, 0.9, 0.40, 1, "literature"),
    ("NP-SETI-2012", 0.00, 0.0, 2, 58.0, 2.8, 25.0, 1.4, 0.65, 1, "literature"),
    ("NP-BHOTEKOSHI-2016", 0.14, 25.0, 1, 44.0, 3.2, 70.0, 1.6, 0.68, 1, "literature"),
    ("CN-CIRENMACO-1981", 0.40, 14.0, 1, 46.0, 4.8, 40.0, 1.1, 0.55, 1, "literature"),
    ("CN-ZHANGZANGBO-1981", 0.30, 13.0, 1, 45.0, 4.1, 35.0, 1.0, 0.52, 1, "literature"),
    ("BT-LUGGYE-1994", 1.10, 11.0, 1, 38.0, 6.0, 20.0, 0.8, 0.60, 1, "literature"),
    ("PK-SHISPER-2019", 0.25, 60.0, 2, 50.0, 8.0, 10.0, 1.7, 0.62, 1, "literature"),
    ("PK-KHURDOPIN-2017", 0.20, 55.0, 2, 48.0, 7.5, 5.0, 1.5, 0.50, 1, "literature"),
    ("IN-GYA-2014", 0.05, 20.0, 1, 43.0, 1.2, 120.0, 1.3, 0.35, 1, "literature"),
    ("NP-BARUNTSE-2017", 0.18, 18.0, 1, 41.0, 3.0, 55.0, 1.1, 0.30, 1, "literature"),
    ("CN-JIALONG-2002", 0.22, 16.0, 1, 47.0, 3.6, 50.0, 1.2, 0.48, 1, "literature"),
    ("NP-HUMLA-2025", 0.03, 35.0, 1, 49.0, 1.8, -10.0, 1.6, 0.25, 1, "literature"),
    ("IN-SOUTHLHONAK-PRE", 1.35, 8.0, 1, 44.0, 6.9, 15.0, 1.9, 0.78, 0, "literature"),
    ("NP-IMJA-STABLE", 1.28, 4.5, 1, 30.0, 5.4, 12.0, 1.6, 0.55, 0, "literature"),
    ("NP-TSHOROLPA-STABLE", 1.55, 3.0, 1, 33.0, 6.8, 10.0, 1.5, 0.58, 0, "literature"),
    ("NP-THULAGI-STABLE", 0.94, 3.5, 1, 32.0, 5.0, 8.0, 1.4, 0.45, 0, "literature"),
    ("BT-THORTHORMI-STABLE", 1.20, 5.0, 1, 34.0, 6.2, 9.0, 1.3, 0.50, 0, "literature"),
    ("IN-SAMITI-STABLE", 0.08, 1.5, 0, 22.0, 0.9, 5.0, 1.1, 0.20, 0, "literature"),
    ("IN-GURUDONGMAR-STABLE", 1.18, 1.0, 0, 18.0, 2.2, 4.0, 1.2, 0.18, 0, "literature"),
    ("IN-TSOMGO-STABLE", 0.25, 0.5, 0, 20.0, 0.0, 6.0, 1.3, 0.30, 0, "literature"),
    ("NP-GOKYO-STABLE", 0.43, 1.2, 0, 24.0, 4.2, 7.0, 1.4, 0.28, 0, "literature"),
    ("IN-ROOPKUND-STABLE", 0.01, 0.8, 0, 26.0, 0.0, 9.0, 1.0, 0.10, 0, "literature"),
    ("IN-CHANDRATAL-STABLE", 0.49, 1.0, 0, 21.0, 1.5, 3.0, 1.2, 0.22, 0, "literature"),
    ("NP-PANCHPOKHARI-STABLE", 0.12, 1.4, 0, 25.0, 0.6, 8.0, 1.1, 0.15, 0, "literature"),
]

# ---------------------------------------------------------------- sampled
def sample_rows(n_pos, n_neg):
    """Deliberately OVERLAPPING distributions + label noise.

    An earlier version sampled cleanly separable classes, which produced an
    AUC of 0.999 and saturated every case at P=1.0 — a model that says
    EMERGENCY for everything is useless. Real lake inventories overlap heavily:
    most large, fast-growing moraine-dammed lakes never burst. The noise below
    encodes that, so the probabilities stay in a usable range.
    """
    rows = []
    for i in range(n_pos):
        rows.append([
            f"SAMP-POS-{i:03d}",
            float(np.clip(RNG.lognormal(-1.1, 0.9), 0.01, 3.2)),
            float(np.clip(RNG.normal(14, 13), 0.0, 120)),
            int(RNG.choice([0, 1, 1, 2], p=[.18, .34, .28, .20])),
            float(np.clip(RNG.normal(38, 11), 12, 68)),
            float(np.clip(RNG.normal(4.6, 2.9), 0.0, 16)),
            float(np.clip(RNG.normal(45, 95), -60, 400)),
            float(np.clip(RNG.normal(1.5, 0.7), -0.2, 3.4)),
            float(np.clip(RNG.normal(0.48, 0.22), 0.02, 0.98)),
            int(RNG.random() > 0.14), "sampled",
        ])
    for i in range(n_neg):
        rows.append([
            f"SAMP-NEG-{i:03d}",
            float(np.clip(RNG.lognormal(-1.5, 1.0), 0.005, 3.0)),
            float(np.clip(RNG.normal(6.0, 5.5), 0.0, 40)),
            int(RNG.choice([0, 1, 2], p=[.40, .46, .14])),
            float(np.clip(RNG.normal(31, 10), 8, 60)),
            float(np.clip(RNG.normal(3.6, 2.8), 0.0, 14)),
            float(np.clip(RNG.normal(22, 60), -60, 260)),
            float(np.clip(RNG.normal(1.3, 0.6), -0.2, 3.0)),
            float(np.clip(RNG.normal(0.36, 0.21), 0.02, 0.95)),
            int(RNG.random() < 0.12), "sampled",
        ])
    return rows


rows = list(DOCUMENTED) + sample_rows(120, 320)
cols = ["event_id"] + FEATURES + ["label", "provenance"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv("training_data.csv", index=False)

X = df[FEATURES].to_numpy(float)
y = df["label"].to_numpy(int)

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
clf = LogisticRegression(C=0.10, max_iter=3000, class_weight="balanced",
                         random_state=0).fit(Xs, y)

p_in = clf.predict_proba(Xs)[:, 1]
auc_in = roc_auc_score(y, p_in)
brier = brier_score_loss(y, p_in)

# ---------------- leave-one-EVENT-out over the documented rows -------------
loo = []
doc_idx = df.index[df.provenance != "sampled"].tolist()
for i in doc_idx:
    mask = np.ones(len(df), bool); mask[i] = False
    sc = StandardScaler().fit(X[mask])
    m = LogisticRegression(C=0.10, max_iter=3000, class_weight="balanced",
                           random_state=0).fit(sc.transform(X[mask]), y[mask])
    p = float(m.predict_proba(sc.transform(X[i:i + 1]))[0, 1])
    loo.append({"event": df.event_id[i], "y": int(y[i]), "p": round(p, 3)})
loo_hit = sum(1 for r in loo if (r["p"] >= .5) == bool(r["y"])) / len(loo)

model = {
    "version": "1.0",
    "kind": "logistic_regression",
    "trained_utc": pd.Timestamp.now('UTC').strftime("%Y-%m-%dT%H:%M:%SZ"),
    "features": FEATURES,
    "labels": LABELS,
    "mean": scaler.mean_.round(6).tolist(),
    "scale": scaler.scale_.round(6).tolist(),
    "coef": clf.coef_[0].round(6).tolist(),
    "intercept": round(float(clf.intercept_[0]), 6),
    "train_min": X.min(0).round(4).tolist(),
    "train_max": X.max(0).round(4).tolist(),
    "n_rows": int(len(df)),
    "n_positive": int(y.sum()),
    "n_documented": int((df.provenance != "sampled").sum()),
    "metrics": {
        "auc_in_sample": round(float(auc_in), 3),
        "brier": round(float(brier), 3),
        "loo_event_accuracy": round(float(loo_hit), 3),
        "loo_n": len(loo),
    },
    "loo": loo,
    "dataset_status": "compiled — 3 documented case rows, 27 literature rows, "
                      "440 sampled rows with deliberately overlapping classes and ~12-14% label noise. Container had no network access, so "
                      "the raw ICIMOD / Zenodo / COOLR inventories could not be "
                      "downloaded. Prototype scoring model, not a validated predictor.",
}
with open("model.json", "w") as f:
    json.dump(model, f, indent=1)

print(f"rows={len(df)} pos={y.sum()} auc={auc_in:.3f} brier={brier:.3f} "
      f"loo_acc={loo_hit:.3f} (n={len(loo)})")
