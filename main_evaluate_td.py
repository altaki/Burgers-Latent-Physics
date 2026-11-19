"""Évaluation TD (expérience « encodeur », prereg outputs/prereg/td_encoder_prereg.json,
commit 515cf5b) — LECTURE SEULE, aucun entraînement, numpy uniquement.

Lit la règle de décision machine-lisible du prereg JSON, charge les
predictions_test.npz des runs td{bras}_{protocole}[_z{mode}]_s{seed}, calcule :
  - métriques primaires à ν=0.01 (heldout_nu) / ν=0.05 (heldout_nu005) :
    Spearman, R² affine, pente, rel_rmse, rang effectif, M6 (définition A1 5623042),
    écart au plafond (R² − 0.9995) ;
  - comparaisons appariées par graine D1 (td2−td0), D2 (td1−td0), D3 (td3−td2,
    EXPLORATOIRE) avec IC unilatéraux 95 % (t ddl=4) et verdicts
    REPRÉSENTATIONNEL / NON REPRÉSENTATIONNEL / INDÉTERMINÉ ;
  - C6'a / C6'b repris de 29c7c3e, par bras (z vs none apparié, n=5 graines).

Tests automatiques : l'ensemble des conditions évaluées doit être EXACTEMENT
l'ensemble du prereg (exit 1 sinon) ; toute valeur NaN évaluée -> exit 1.

Sortie : outputs/logs/td_encoder_results.json
Usage : python3 main_evaluate_td.py
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

import td_protocols as tp  # source de vérité unique des protocoles (correction bug filtre)

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
MODELS = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"
LOGS = REPO / "outputs" / "logs"
PREREG = REPO / "outputs" / "prereg" / "td_encoder_prereg.json"

# quantiles unilatéraux 95 % de la loi de Student
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015}
PLAFOND_R2 = 0.9995  # backward_w25 à ν=0.01 (experiment_a_seq_analytic_baseline.json)
PCA_K8 = 0.188       # plancher PCA k=8 (prereg 29c7c3e)


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


def effective_rank(h, var_threshold=0.95):
    """Rang effectif 95 % (ml_models.latent_pca_stats:574-599)."""
    X = h - h.mean(axis=0)
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    ev = np.linalg.eigvalsh(C)[::-1]
    tot = ev.sum()
    if tot <= 1e-14:
        return 0.0
    cum = np.cumsum(ev) / tot
    return float(min(int(np.searchsorted(cum, var_threshold) + 1), ev.size))


def participation_ratio(h):
    X = h - h.mean(axis=0)
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    ev = np.linalg.eigvalsh(C)[::-1]
    tot = ev.sum()
    return float(tot * tot / np.sum(ev ** 2)) if tot > 1e-14 else 0.0


def run_metrics(npz_path: Path, nu_eval: float, nuD_traj: dict) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    nu = d["nu"].astype(float)
    m = np.isclose(nu, nu_eval)
    z = d["z"].astype(float)[m]
    y = d["nu_dissipation_t"].astype(float)[m]
    D = d["D"].astype(float)[m]
    r2 = 1 - np.sum((z - y) ** 2) / np.sum((y - y.mean()) ** 2)
    slope = float(np.sum((y - y.mean()) * (z - z.mean())) / np.sum((z - z.mean()) ** 2))
    rel = float(np.sqrt(np.mean((z - y) ** 2)) / y.mean())
    out = {
        "spearman_z_nuD": spearman(z, y),
        "r2_affine_nuD_z": float(r2),
        "calibration_slope": slope,
        "rel_rmse_z": rel,
        "ecart_au_plafond": float(r2 - PLAFOND_R2),
        "effective_rank_h": effective_rank(d["h"].astype(float)) if d["h"].size else float("nan"),
        "participation_ratio_h": participation_ratio(d["h"].astype(float)) if d["h"].size else float("nan"),
        "median_z_over_D": float(np.median(z / D)),
        "n_windows": int(m.sum()),
        "M6_raidissement": m6_raidissement(d, m, nu_eval, nuD_traj),
    }
    # rec RMSE depuis le metrics JSON du run
    run_dir = npz_path.parent
    zmode = str(d["z_target_mode"])
    proto = str(d["protocol"])
    bras = run_dir.name.split("_")[0]  # td0..td3
    suffix = f"{bras}_{proto}" + (f"_z{zmode}" if zmode != "none" else "")
    mj = run_dir / f"metrics_window_a_{suffix}.json"
    if mj.exists():
        r = json.load(open(mj))
        out["reconstruction_rmse_test"] = float(r.get("reconstruction_rmse_test", float("nan")))
    else:
        out["reconstruction_rmse_test"] = float("nan")
    return out


def m6_raidissement(d, mask, nu_eval, nuD_traj) -> float:
    """M6 (prereg A1 5623042) : raidissement = t < t_pic, t_pic = argmax global de
    νD_vrai sur [0,100] ; M6 = (max−min z)/(max−min νD_vrai) sur les pas en
    raidissement ; trajectoires avec < 5 pas en raidissement exclues ; médiane."""
    traj = d["trajectory_index"].astype(int)[mask]
    tidx = d["time_index_t"].astype(int)[mask]
    z = d["z"].astype(float)[mask]
    nuD = d["nu_dissipation_t"].astype(float)[mask]
    per_traj = []
    for ti in np.unique(traj):
        sel = traj == ti
        if ti not in nuD_traj:
            continue
        nuD_full = nuD_traj[ti]  # (101,)
        t_pic = int(np.argmax(nuD_full))
        raid = sel & (tidx < t_pic)
        if raid.sum() < 5:
            continue
        zr, yr = z[raid], nuD[raid]
        denom = yr.max() - yr.min()
        if denom <= 0:
            continue
        per_traj.append(float((zr.max() - zr.min()) / denom))
    return float(np.median(per_traj)) if per_traj else float("nan")


def uni_ic(vals, side):
    """IC unilatéral 95 % : side='lower' -> [m - t*sd/√n, +inf) ; 'upper' -> (-inf, m + t*sd/√n]."""
    v = np.asarray(vals, float)
    n = v.size
    m = float(np.mean(v))
    if n < 2:
        return m, float("nan")
    sd = float(np.std(v, ddof=1))
    t = T95.get(n - 1, 1.645)
    half = t * sd / sqrt(n)
    return m, (m - half if side == "lower" else m + half)


def main() -> int:
    prereg = json.load(open(PREREG))
    bras_ids = [b["id"] for b in prereg["bras"]]
    seeds = prereg["seeds"]
    protocoles = {p["name"]: p for p in prereg["protocoles"]}
    nu_eval = {name: float(p["nu_eval_principal"]) for name, p in protocoles.items()}
    # cohérence module de protocoles unique vs prereg (correction bug filtre)
    tp.check_against_prereg()
    if set(tp.PROTOCOLS) != set(protocoles):
        print("ÉCHEC : protocoles du module != prereg")
        return 1
    delta_R = float(prereg["decision_rule"]["comparaison_principale"]["delta_R"])

    # νD_vrai par trajectoire du test (pour M6) : dissipation_physical (101,)
    te = np.load(DATA / "test_data_nu_sweep.npz", allow_pickle=True)
    nuD_traj = {int(i): te["dissipation_physical"][i].astype(float) for i in range(te["states"].shape[0])}

    # découverte des runs
    runs = {}
    for bras in bras_ids:
        for proto in protocoles:
            for mode in ("energy_centered", "none"):
                for seed in seeds:
                    suffix = f"{bras}_{proto}" + ("_zenergy_centered" if mode == "energy_centered" else "")
                    rdir = MODELS / f"{suffix}_s{seed}"
                    p = rdir / "predictions_test.npz"
                    entry = {"bras": bras, "protocole": proto, "mode": mode, "seed": seed,
                             "run": f"{suffix}_s{seed}", "present": p.exists()}
                    if p.exists():
                        mm = run_metrics(p, nu_eval[proto], nuD_traj)
                        mm.update({"seed": seed, "run": rdir.name})
                        entry.update(mm)
                    runs[(bras, proto, mode, seed)] = entry

    # runs manquants -> exit 1 (aucune condition ne peut être évaluée sans ses runs)
    missing = [e["run"] for e in runs.values() if not e.get("present")]
    if missing:
        print(f"ÉCHEC : {len(missing)} runs manquants (attendus {len(runs)}) :", missing[:10])
        return 1

    # test d'égalité des conditions évaluées vs prereg + NaN -> exit 1
    expected_conds = {"D1", "D2", "D3"} | {f"C6a_{b}" for b in bras_ids} | {f"C6b_{b}" for b in bras_ids}
    evaluated = set()
    nan_hits = []
    metric_keys = ["spearman_z_nuD", "r2_affine_nuD_z", "calibration_slope", "rel_rmse_z",
                   "effective_rank_h", "reconstruction_rmse_test", "M6_raidissement"]
    for key, e in runs.items():
        for mk in metric_keys:
            if mk in e and (not np.isfinite(e[mk])):
                nan_hits.append((e["run"], mk, e[mk]))
    if nan_hits:
        print("ÉCHEC : NaN/inf dans les métriques évaluées :", nan_hits[:10])
        return 1

    results = {"experiment": "TD_encoder_inductive_bias", "prereg": str(PREREG),
               "prereg_git_note": "commit 515cf5b", "runs": {}, "comparisons": {}, "c6_prime": {}}

    # métriques par run (protocole principal en premier)
    for proto in tp.PROTOCOLS:
        for bras in bras_ids:
            for mode in ("energy_centered", "none"):
                rows = [runs[(bras, proto, mode, s)] for s in seeds]
                results["runs"][f"{bras}_{proto}_{mode}"] = [
                    {k: v for k, v in r.items() if k != "present"} for r in rows]

    def per_seed(bras, proto, mode, metric):
        return {s: runs[(bras, proto, mode, s)][metric] for s in seeds
                if runs[(bras, proto, mode, s)].get("present")}

    # comparaisons appariées D1/D2/D3 (protocole principal, mode energy_centered)
    proto_main = "heldout_nu"
    for cid, hi, lo in (("D1", "td2", "td0"), ("D2", "td1", "td0"), ("D3", "td3", "td2")):
        d = {}
        for s in seeds:
            a = runs.get((hi, proto_main, "energy_centered", s))
            b = runs.get((lo, proto_main, "energy_centered", s))
            if a and b and a.get("present") and b.get("present"):
                d[s] = a["r2_affine_nuD_z"] - b["r2_affine_nuD_z"]
        vals = list(d.values())
        m, lower = uni_ic(vals, "lower")
        _, upper = uni_ic(vals, "upper")
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        if cid == "D1":
            if m >= delta_R and lower > 0:
                verdict = "REPRESENTATIONNEL"
            elif upper < delta_R:
                verdict = "NON REPRESENTATIONNEL"
            else:
                verdict = "INDETERMINE"
        else:
            verdict = "INDETERMINE" if not (m >= delta_R and lower > 0) and not (upper < delta_R) else (
                "REPRESENTATIONNEL" if (m >= delta_R and lower > 0) else "NON REPRESENTATIONNEL")
        results["comparisons"][cid] = {
            "bras_haut": hi, "bras_bas": lo, "metric": "r2_affine_nuD_z", "nu": nu_eval[proto_main],
            "protocole": proto_main, "mode": "energy_centered", "n": len(vals),
            "per_seed": {str(k): v for k, v in d.items()},
            "mean": m, "sd": sd, "lower_unil95": lower, "upper_unil95": upper,
            "delta_R": delta_R, "verdict": verdict,
            "statut": "CONFIRMATOIRE" if cid in ("D1", "D2") else "EXPLORATOIRE",
        }
        evaluated.add(cid)

    # C6'a / C6'b par bras (protocole principal, n=5)
    for bras in bras_ids:
        rec_z = per_seed(bras, proto_main, "energy_centered", "reconstruction_rmse_test")
        rec_n = per_seed(bras, proto_main, "none", "reconstruction_rmse_test")
        common = [s for s in seeds if s in rec_z and s in rec_n]
        ratios = {s: rec_z[s] / rec_n[s] for s in common}
        rv = list(ratios.values())
        m, upper = uni_ic(rv, "upper")
        c6a = upper < 1.10
        rec_z_vals = [rec_z[s] for s in common]
        m_rec, _ = uni_ic(rec_z_vals, "upper")
        c6b = m_rec <= PCA_K8
        results["c6_prime"][bras] = {
            "protocole": proto_main, "n": len(common),
            "ratios_per_seed": {str(k): v for k, v in ratios.items()},
            "mean_ratio": m, "upper_unil95_ratio": upper, "delta": 0.10,
            "C6a": bool(c6a), "rec_z_mean": m_rec, "C6b": bool(c6b), "plancher_k8": PCA_K8,
        }
        evaluated.add(f"C6a_{bras}")
        evaluated.add(f"C6b_{bras}")

    if evaluated != expected_conds:
        print("ÉCHEC : conditions évaluées != prereg")
        print("  manquantes :", sorted(expected_conds - evaluated))
        print("  en trop :", sorted(evaluated - expected_conds))
        return 1

    results["verdict_principal_D1"] = results["comparisons"]["D1"]["verdict"]
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "td_encoder_results.json"
    json.dump(results, open(out, "w"), indent=1, ensure_ascii=False, default=str)
    print(f"verdict D1 (td2-td0, nu=0.01 held-out) : {results['verdict_principal_D1']}")
    print(f"résultats écrits : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
