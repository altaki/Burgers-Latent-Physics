"""Reproduit la stratification « phase_strat » du Step 6 (aphysis_stratifications.json),
dont le code générateur n'était pas versionné.

Définition reconstituée et vérifiée numériquement :
  - fenêtres A1 : t_index ∈ [24, 99] (w=25, stride=1) ;
  - phase = dD/dt > 0 avec dD/dt = np.diff(D, axis=1) (k=1), D = clé `dissipation` du npz test ;
  - frac_dDdt_gt0 AFFICHÉE = n(dD/dt>0 à ν=0.01) / n_total_poolé(3 strates) = 2514/11400 = 22.05 %
    (dénominateur poolé ; la fraction intra-strate ν=0.01 est 2514/3800 = 66.16 %) ;
  - métriques (rel_rmse, r2, spearman, slope) calculées sur les fenêtres de la strate ν=0.01
    selon la phase, z vs nu_dissipation_t.

Lecture seule. Sortie : outputs/logs/phase_strat_reproduction.json
Usage : python3 main_reproduce_phase_strat.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
MODELS = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"
LOGS = REPO / "outputs" / "logs"

T_LO, T_HI = 24, 100  # t_index ∈ [24, 99]


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
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def main():
    truth = np.load(DATA / "test_data_nu_sweep.npz", allow_pickle=True)
    D = truth["dissipation"].astype(float)
    nu = truth["nu_values"].astype(float)
    t_idx = np.arange(T_LO, T_HI)
    dDdt = np.diff(D, axis=1)[:, t_idx] > 0          # (150, 76)
    n_total = int(dDdt.size)
    frac_strate = {}
    for v in (0.01, 0.05, 0.2):
        m = np.isclose(nu, v)
        frac_strate[f"nu={v}"] = {"n_gt0": int(dDdt[m].sum()), "n": int(dDdt[m].size),
                                  "frac": float(dDdt[m].mean())}
    n_gt0_001 = frac_strate["nu=0.01"]["n_gt0"]

    ref = json.load(open(LOGS / "aphysis_stratifications.json"))
    out = {
        "definition": "dD/dt = np.diff(D, axis=1) > 0 sur t_index ∈ [24,99] ; D = clé 'dissipation'",
        "frac_dDdt_gt0_affichee": n_gt0_001 / n_total,
        "n_gt0_nu001": n_gt0_001, "n_total_poolé": n_total,
        "frac_intra_strate": frac_strate,
        "runs": {},
    }
    for run_dir in sorted(p.name for p in MODELS.iterdir() if (p / "predictions_test.npz").exists()):
        p = np.load(MODELS / run_dir / "predictions_test.npz", allow_pickle=True)
        ti = p["trajectory_index"].astype(int); tt = p["time_index_t"].astype(int)
        nuv = p["nu"].astype(float); zv = p["z"].astype(float); ndv = p["nu_dissipation_t"].astype(float)
        m01 = np.isclose(nuv, 0.01)
        phase = dDdt[ti[m01], tt[m01] - T_LO]
        z = zv[m01]; y = ndv[m01]
        res = {}
        for name, mask in (("dDdt_gt0", phase), ("dDdt_lt0", ~phase)):
            zz, yy = z[mask], y[mask]
            rel = float(np.sqrt(np.mean((zz - yy) ** 2)) / yy.mean())  # rel_rmse = rmse / mean(nuD), vérifié vs publié
            r2 = 1 - np.sum((zz - yy) ** 2) / np.sum((yy - yy.mean()) ** 2)
            slope = float(np.sum((yy - yy.mean()) * (zz - zz.mean())) / np.sum((yy - yy.mean()) ** 2))
            res[name] = {"n": int(mask.sum()), "rel_rmse": float(rel), "r2": float(r2),
                         "spearman": spearman(zz, yy), "slope": slope}
        res["frac_dDdt_gt0"] = float(phase.mean())
        # comparaison au json de référence si présent
        key = run_dir.replace("window_w25_heldout_nu", "").replace("window_w25_standard", "") or "s42"
        out["runs"][run_dir] = res

    # comparaison directe avec aphysis_stratifications.json (clés none_s42 / aphys_s42 / ...)
    mapping = {
        "window_w25_heldout_nu": "none_s42", "window_w25_heldout_nu_s7": "none_s7",
        "window_w25_heldout_nu_s123": "none_s123",
        "window_w25_heldout_nu_zenergy_centered_s42": "aphys_s42",
        "window_w25_heldout_nu_zenergy_centered_s7": "aphys_s7",
        "window_w25_heldout_nu_zenergy_centered_s123": "aphys_s123",
        "window_w25_heldout_nu_zsupervised_s42": "asup_s42",
    }
    comp = {}
    for run_dir, tag in mapping.items():
        if tag not in ref or run_dir not in out["runs"]:
            continue
        r = ref[tag]["phase_strat_nu001"]
        mine = out["runs"][run_dir]
        comp[tag] = {
            "frac_publie": r["frac_dDdt_gt0"], "frac_reproduite_intra_strate": mine["frac_dDdt_gt0"],
            "frac_reproduite_affichee_poolee": out["frac_dDdt_gt0_affichee"],
            "n_gt0_publie": r["dDdt_gt0"]["n"], "n_gt0_reproduit": mine["dDdt_gt0"]["n"],
            "rel_rmse_gt0_publie": r["dDdt_gt0"]["rel_rmse"], "rel_rmse_gt0_reproduit": mine["dDdt_gt0"]["rel_rmse"],
            "rel_rmse_lt0_publie": r["dDdt_lt0"]["rel_rmse"], "rel_rmse_lt0_reproduit": mine["dDdt_lt0"]["rel_rmse"],
        }
    out["comparaison_aphysis_stratifications"] = comp

    with open(LOGS / "phase_strat_reproduction.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1)[:1800])
    print("runs reproduits:", len(out["runs"]))


if __name__ == "__main__":
    main()
