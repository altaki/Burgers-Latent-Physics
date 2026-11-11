"""Évaluation B1 (interpolation held-out nu=0.05) — LECTURE SEULE, aucun entraînement.

Lit la règle de décision machine-lisible de outputs/prereg/b1_interpolation_prereg.json
(commit f431e66), charge les predictions_test.npz des runs window_w25_heldout_nu005*,
calcule les métriques à nu=0.05, évalue C1-C7 + F1, produit le verdict.

Test automatique : l'ensemble des conditions évaluées doit être EXACTEMENT l'ensemble
des conditions du JSON (C1-C7 + F1) — échec (exit 1) sinon.

Sortie : outputs/logs/b1_interpolation_results.json
Usage : python3 main_evaluate_b1.py
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
MODELS = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"
LOGS = REPO / "outputs" / "logs"
PREREG = REPO / "outputs" / "prereg" / "b1_interpolation_prereg.json"

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def rankdata(a):
    a = np.asarray(a, float)
    s = np.argsort(a, kind="mergesort")
    inv = np.empty_like(s); inv[s] = np.arange(len(a))
    as_ = a[s]
    obs = np.r_[True, as_[1:] != as_[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


def t_ic95(vals):
    v = np.asarray(vals, float)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(v))
    if n < 2:
        return m, float("nan"), float("nan")
    sd = float(np.std(v, ddof=1))
    half = T975.get(n - 1, 1.96) * sd / sqrt(n)
    return m, m - half, m + half


def effective_rank(h):
    """Participation ratio des valeurs propres de la PCA de h (n, d)."""
    X = h - h.mean(axis=0)
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (ev ** 2).sum()) if ev.size else float("nan")


def run_metrics_nu005(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    nu = d["nu"].astype(float)
    m = np.isclose(nu, 0.05)
    z = d["z"].astype(float)[m]
    y = d["nu_dissipation_t"].astype(float)[m]
    D = d["dissipation_t"].astype(float)[m] if "dissipation_t" in d.files else None
    r2 = 1 - np.sum((z - y) ** 2) / np.sum((y - y.mean()) ** 2)
    slope = float(np.sum((y - y.mean()) * (z - z.mean())) / np.sum((y - y.mean()) ** 2))
    rel = float(np.sqrt(np.mean((z - y) ** 2)) / y.mean())
    rank = effective_rank(d["h"].astype(float)) if d["h"].size else float("nan")
    out = {
        "spearman_z_nuD": spearman(z, y),
        "r2_affine_nuD_z": float(r2),
        "calibration_slope": slope,
        "rel_rmse_z": rel,
        "effective_rank_h": rank,
        "median_z_over_D": float(np.median(z / D)) if D is not None else float("nan"),
        "n_windows_nu005": int(m.sum()),
    }
    # rec RMSE depuis le metrics json du run si présent
    mj = npz_path.parent / f"metrics_window_a_w25_heldout_nu005{'_z' + str(d['z_target_mode']) if str(d['z_target_mode']) != 'none' else ''}.json"
    if mj.exists():
        r = json.load(open(mj))
        out["reconstruction_rmse_test"] = float(r.get("reconstruction_rmse_test", float("nan")))
    return out


def main():
    prereg = json.load(open(PREREG))
    nu_eval = float(prereg["nu_eval"])
    rule = prereg["decision_rule"]
    conds = rule["conditions"]
    fail = rule["fail_condition"]

    # nu_oracle : moyenne globale de nuD sur le train filtre (nu in {0.01,0.20})
    tr = np.load(DATA / "train_data_nu_sweep.npz", allow_pickle=True)
    nu_tr = tr["nu_values"].astype(float)
    mtr = np.isclose(nu_tr, 0.01) | np.isclose(nu_tr, 0.2)
    t_idx = np.arange(24, 100)
    nuD_tr = (nu_tr[:, None] * tr["dissipation"].astype(float))[mtr][:, t_idx].ravel()
    oracle_val = float(nuD_tr.mean())
    te = np.load(DATA / "test_data_nu_sweep.npz", allow_pickle=True)
    nu_te = te["nu_values"].astype(float)
    m05 = np.isclose(nu_te, 0.05)
    nuD_te_005 = (nu_te[:, None] * te["dissipation"].astype(float))[m05][:, t_idx].ravel()
    oracle_rel = float(np.sqrt(np.mean((oracle_val - nuD_te_005) ** 2)) / nuD_te_005.mean())

    # runs B1
    runs = {}
    for tag_info in prereg["runs"]:
        tag = tag_info["tag"]
        runs[tag] = []
        for seed in tag_info["seeds"]:
            pat = tag_info["run_pattern"].replace("{seed}", str(seed))
            p = MODELS / pat / "predictions_test.npz"
            if not p.exists():
                runs[tag].append({"seed": seed, "run": pat, "present": False})
                continue
            mm = run_metrics_nu005(p)
            mm.update({"seed": seed, "run": pat, "present": True})
            runs[tag].append(mm)

    # évaluation des conditions sur la moyenne 3 graines du bras aphys_mono0
    aphys = [r for r in runs["aphys_mono0"] if r.get("present")]
    none = [r for r in runs["none"] if r.get("present")]

    def agg(metric):
        vals = [r[metric] for r in aphys if metric in r]
        m, lo, hi = t_ic95(vals)
        return {"mean": m, "ic95": [lo, hi], "per_seed": vals}

    eval_conditions = {}
    for c in conds:
        cid, metric, op = c["id"], c["metric"], c["op"]
        if metric == "rel_rmse_z" and cid == "C4_beats_nu_oracle":
            a = agg("rel_rmse_z")
            val = a["mean"] < oracle_rel
            eval_conditions[cid] = {"value": a["mean"], "comparator": oracle_rel, "result": bool(val), "ic95": a["ic95"]}
        else:
            a = agg(metric)
            v = a["mean"]
            if op == ">=":
                res = v >= c["threshold"]
            elif op == "<=":
                res = v <= c["threshold"]
            elif op == "in":
                res = c["range"][0] <= v <= c["range"][1]
            else:
                res = False
            eval_conditions[cid] = {"value": v, "ic95": a["ic95"], "op": op,
                                     "threshold": c.get("threshold", c.get("range")), "result": bool(res)}
    # fail condition
    a = agg(fail["metric"])
    eval_conditions[fail["id"]] = {"value": a["mean"], "ic95": a["ic95"], "op": fail["op"],
                                   "threshold": fail["threshold"],
                                   "result": bool(a["mean"] < fail["threshold"])}

    # TEST AUTOMATIQUE : ensemble évalué == ensemble du JSON
    json_ids = {c["id"] for c in conds} | {fail["id"]}
    eval_ids = set(eval_conditions.keys())
    test_ok = json_ids == eval_ids
    if not test_ok:
        print("ÉCHEC test d'égalité : manquants =", json_ids - eval_ids, " en trop =", eval_ids - json_ids)
        sys.exit(1)

    all_ok = all(eval_conditions[c["id"]]["result"] for c in conds)
    if not aphys:
        verdict = "EN_ATTENTE_RUNS"
    elif all_ok:
        verdict = "IDENTIFIABLE"
    elif eval_conditions[fail["id"]]["result"]:
        verdict = "NOT IDENTIFIABLE"
    else:
        verdict = "PARTIAL"

    # complémentaire : appariement vs none
    pair = {}
    if none:
        for r in aphys:
            rn = next((x for x in none if x["seed"] == r["seed"]), None)
            if rn:
                pair[f"s{r['seed']}"] = {"d_spearman": r["spearman_z_nuD"] - rn["spearman_z_nuD"],
                                         "d_r2": r["r2_affine_nuD_z"] - rn["r2_affine_nuD_z"]}

    out = {
        "prereg": str(PREREG.relative_to(REPO)),
        "prereg_commit": "f431e66afce1b42ee90e5424ac0168c342ef85ed",
        "nu_eval": nu_eval,
        "nu_oracle": {"valeur": oracle_val, "rel_rmse_nu005": oracle_rel},
        "runs": runs,
        "conditions_evaluees": eval_conditions,
        "test_egalite_ensembles": test_ok,
        "verdict": verdict,
        "complementaire_appariement_vs_none": pair,
        "note": "appariement vs none = COMPLEMENTAIRE (hors verdict), leçon A1 ccb139a",
    }
    with open(LOGS / "b1_interpolation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print("verdict:", verdict, "| test_egalite:", test_ok)
    for cid, e in eval_conditions.items():
        print(f"  {cid}: value={e['value']:.4f} result={e['result']}")
    print("->", LOGS / "b1_interpolation_results.json")


if __name__ == "__main__":
    main()
