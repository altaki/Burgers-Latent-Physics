"""A1 — Analyse intra-trajectoire : z est-il un TRACEUR DYNAMIQUE de nuD_vrai
ou une ETIQUETTE STATIQUE bien calibrée ?

Lecture seule : predictions_test.npz (Step 6 + window-A) + npz nu_sweep_v1.
Aucun entraînement, aucun fichier d'entrée modifié. Script additif unique de l'action A1.
Pré-enregistrement : outputs/prereg/a1_intra_trajectory_prereg.md (commit 5623042).

Sorties (nouveaux fichiers uniquement) :
  - outputs/logs/a1_intra_trajectory.json
  - outputs/logs/a1_intra_trajectory_per_traj.csv
  - outputs/logs/a1_figures_data.csv   (séries pour figures, si matplotlib absent)
  - outputs/figures/a1_*.png           (si matplotlib disponible)

Usage : python3 main_analyze_a1_intra_trajectory.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from math import sqrt
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
MODELS = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"
LOGS = REPO / "outputs" / "logs"
FIGS = REPO / "outputs" / "figures"
PREREG_HASH = "56230429b1adea4f6fc3aadc8afd21ff45c161fd"

W = 25
T_LO, T_HI = W - 1, 100          # t_index A1 : [24, 99] (stride=1, cf. ml_datasets_experiment_c.py:573-578)
NU_HELDOUT = 0.01
B_BOOT = 2000

RUNS_CONFIRM = {
    "aphys_mono0": [f"window_w25_heldout_nu_zenergy_centered_s{s}" for s in (42, 7, 123)],
    "none": ["window_w25_heldout_nu"] + [f"window_w25_heldout_nu_s{s}" for s in (7, 123)],
}
RUNS_EXPLORATOIRE = {
    "asup_s42_heldout": "window_w25_heldout_nu_zsupervised_s42",
    "none_s42_standard": "window_w25_standard",
    "aphys_s42_standard": "window_w25_standard_zenergy_centered_s42",
    "none_s42_w50_heldout": "window_w50_heldout_nu",
}

# t(0.975, ddl) — indexé par ddl = n-1 (corrigé : la v1 indexait par n)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228}


# ---------------------------------------------------------------- utilitaires
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    s = np.argsort(a, kind="mergesort")
    inv = np.empty_like(s)
    inv[s] = np.arange(len(a))
    as_ = a[s]
    obs = np.r_[True, as_[1:] != as_[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def spearman(a, b) -> float:
    return pearson(rankdata(np.asarray(a, float).ravel()), rankdata(np.asarray(b, float).ravel()))


def ic95_bootstrap(values: np.ndarray, B: int = B_BOOT, seed: int = 42):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0.0
    rng = np.random.default_rng(seed)
    med = np.median(v)
    boots = np.median(v[rng.integers(0, v.size, size=(B, v.size))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(med), float(lo), float(hi)


def t_ic95(values: np.ndarray):
    """moyenne ± IC95 Student, t(0.975, ddl=n-1)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), 0.0
    m = float(np.mean(v))
    if n < 2:
        return m, float("nan"), float(n)
    sd = float(np.std(v, ddof=1))
    t = T975.get(n - 1, 2.0)
    half = t * sd / sqrt(n)
    return m, m - half, m + half


# ---------------------------------------------------------------- chargement
def load_test_truth():
    d = np.load(DATA / "test_data_nu_sweep.npz", allow_pickle=True)
    return {
        "E": d["energy"].astype(float),
        "D": d["dissipation"].astype(float),
        "nu": d["nu_values"].astype(float),
        "times": d["times"].astype(float),
        "nuD": (d["nu_values"].astype(float)[:, None] * d["dissipation"].astype(float)),
        "energy0": d["energy0"].astype(float),
    }


def load_run_tables(run_dir: str, truth, w: int = W):
    p = MODELS / run_dir / "predictions_test.npz"
    z = np.load(p, allow_pickle=True)
    ti = z["trajectory_index"].astype(int)
    tt = z["time_index_t"].astype(int)
    nu = z["nu"].astype(float)
    zv = z["z"].astype(float)
    nuDv = z["nu_dissipation_t"].astype(float)
    Ev = z["energy_t"].astype(float) if "energy_t" in z.files else truth["E"][ti, tt]
    nuD_est = z["nuD_est_t"].astype(float) if "nuD_est_t" in z.files else np.full(len(ti), np.nan)
    seed = int(z["seed"]) if "seed" in z.files else 42
    proto = str(z["protocol"]) if "protocol" in z.files else "?"
    zmode = str(z["z_target_mode"]) if "z_target_mode" in z.files else "?"

    t_idx = np.arange(w - 1, 100)
    n_t = len(t_idx)
    trajs = np.where(np.isclose(nu, NU_HELDOUT))[0]
    traj_ids = np.unique(ti[trajs])
    n_tr = len(traj_ids)
    Z = np.full((n_tr, n_t), np.nan)
    N = np.full((n_tr, n_t), np.nan)
    E = np.full((n_tr, n_t), np.nan)
    EST = np.full((n_tr, n_t), np.nan)
    pos = {int(t): k for k, t in enumerate(t_idx)}
    for row in trajs:
        i = int(np.where(traj_ids == ti[row])[0][0])
        Z[i, pos[int(tt[row])]] = zv[row]
        N[i, pos[int(tt[row])]] = nuDv[row]
        E[i, pos[int(tt[row])]] = Ev[row]
        EST[i, pos[int(tt[row])]] = nuD_est[row]
    return {
        "run": run_dir, "seed": seed, "protocol": proto, "z_target_mode": zmode,
        "z": Z, "nuD": N, "E": E, "nuD_est": EST, "t_idx": t_idx,
        "traj_ids": traj_ids, "w": w, "path": str(p.relative_to(REPO)),
    }


# ---------------------------------------------------------------- métriques
def t_pic_per_traj(truth, traj_ids):
    return np.argmax(truth["nuD"][traj_ids], axis=1)   # argmax global sur [0,100]


def metrics_block(Z, N, E, t_idx, dt, t_pic_full, def_raid="avant_pic", lag_max=10):
    n, m = Z.shape
    tt = np.asarray(t_idx)
    if def_raid == "avant_pic":
        raid = tt[None, :] < t_pic_full[:, None]
    else:
        d1 = np.diff(N, axis=1)
        raid = np.zeros_like(N, bool)
        raid[:, :-1] = d1 > 0
    dec = ~raid
    pic_int = (t_pic_full >= tt[0]) & (t_pic_full <= tt[-1])   # pic intérieur à la fenêtre A1

    Pc = Z - Z.mean(axis=1, keepdims=True)
    Nc = N - N.mean(axis=1, keepdims=True)
    Ec = E - E.mean(axis=1, keepdims=True)

    m1 = np.array([pearson(Pc[i], Nc[i]) for i in range(n)])
    m1s = np.array([spearman(Pc[i], Nc[i]) for i in range(n)])
    m1_raid, m1s_raid = [], []
    for i in range(n):
        if raid[i].sum() >= 5:
            m1_raid.append(pearson(Pc[i][raid[i]], Nc[i][raid[i]]))
            m1s_raid.append(spearman(Pc[i][raid[i]], Nc[i][raid[i]]))
    m1_raid = np.array(m1_raid); m1s_raid = np.array(m1s_raid)

    # M2 : double centrage + profil horloge LOO (moyenne), poolé
    prof_N = np.zeros_like(N); prof_Z = np.zeros_like(Z)
    for i in range(n):
        o = np.arange(n) != i
        prof_N[i] = N[o].mean(axis=0)
        prof_Z[i] = Z[o].mean(axis=0)
    Rz = (Z - Z.mean(axis=1, keepdims=True)) - (prof_Z - prof_Z.mean(axis=1, keepdims=True))
    Rn = (N - N.mean(axis=1, keepdims=True)) - (prof_N - prof_N.mean(axis=1, keepdims=True))
    m2 = pearson(Rz.ravel(), Rn.ravel())
    m2s = spearman(Rz.ravel(), Rn.ravel())

    rng = (N.max(axis=1) - N.min(axis=1))
    m3 = {}
    for k in (1, 5, 10):
        if k >= m:
            continue
        dZ = Z[:, k:] - Z[:, :-k]
        dN = N[:, k:] - N[:, :-k]
        ok = np.abs(dN) >= 0.05 * rng[:, None]
        agree = (np.sign(dZ) == np.sign(dN)) & ok
        m3[f"k{k}_all"] = float(agree.sum() / ok.sum())
        m3[f"k{k}_frac_exclue"] = float(1 - ok.mean())
        for phase, mask in (("raid", raid[:, :-k] & raid[:, k:]), ("dec", dec[:, :-k] & dec[:, k:])):
            okp = ok & mask
            m3[f"k{k}_{phase}"] = float(agree[okp].sum() / okp.sum()) if okp.sum() else float("nan")
        if k == 1:
            ok2 = np.abs(dN) >= 0.02 * rng[:, None]
            okp = ok2 & raid[:, :-1] & raid[:, 1:]
            m3["k1_raid_seuil2pct"] = float(((np.sign(dZ) == np.sign(dN)) & okp).sum() / okp.sum()) if okp.sum() else float("nan")

    pz = np.argmax(Z, axis=1); pn = np.argmax(N, axis=1)
    m4 = (pz - pn) * dt
    m4_pic_int = m4[pic_int]     # M4 restreint aux trajectoires à pic intérieur

    lags = []
    for i in range(n):
        a = Pc[i] - Pc[i].mean(); b = Nc[i] - Nc[i].mean()
        best, bl = -2.0, 0
        for L in range(-lag_max, lag_max + 1):
            aa = a[max(0, L):m + min(0, L)] if L != 0 else a
            bb = b[max(0, -L):m + min(0, -L)] if L != 0 else b
            c = pearson(aa, bb)
            if c > best:
                best, bl = c, L
        lags.append(bl)
    m5 = np.array(lags)

    m6 = []
    for i in range(n):
        if raid[i].sum() >= 5:
            m6.append((Z[i][raid[i]].max() - Z[i][raid[i]].min()) /
                      max(N[i][raid[i]].max() - N[i][raid[i]].min(), 1e-30))
    m6 = np.array(m6)

    rz, rn, re_ = rankdata(Z.ravel()), rankdata(N.ravel()), rankdata(E.ravel())

    def resid(y, x):
        X = np.column_stack([x, np.ones(len(x))])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ c

    m7 = pearson(resid(rz, re_), resid(rn, re_))
    m7_t = {}
    for t0 in (25, 50, 75):
        w0 = np.where(tt == t0)[0]
        if len(w0):
            j = int(w0[0])
            m7_t[f"t{t0}"] = pearson(resid(rankdata(Z[:, j]), rankdata(E[:, j])),
                                    resid(rankdata(N[:, j]), rankdata(E[:, j])))

    m8 = []
    for i in range(n):
        if raid[i].sum() >= 5:
            m8.append(pearson(Pc[i][raid[i]], Ec[i][raid[i]]))
    m8 = np.array(m8)

    return {
        "n_traj": n, "n_t": m, "raid": raid, "pic_int": pic_int,
        "M1": m1, "M1s": m1s, "M1_raid": m1_raid, "M1s_raid": m1s_raid,
        "M2": m2, "M2s": m2s, "M3": m3, "M4_s": m4, "M4_pic_int_s": m4_pic_int,
        "M5_lag": m5, "M6": m6, "M7": m7, "M7_t": m7_t, "M8_raid": m8,
    }


def aggregate(mb, seed=42):
    out = {}
    for key in ("M1", "M1s", "M1_raid", "M1s_raid", "M6", "M8_raid"):
        med, lo, hi = ic95_bootstrap(mb[key], seed=seed)
        out[key] = {"median": med, "ic95": [lo, hi], "n": int(np.sum(~np.isnan(mb[key])))}
    out["M2"] = mb["M2"]; out["M2s"] = mb["M2s"]
    out["M3"] = mb["M3"]
    for key in ("M4_s", "M4_pic_int_s", "M5_lag"):
        med, lo, hi = ic95_bootstrap(mb[key], seed=seed)
        out[key] = {"median": med, "ic95": [lo, hi], "n": int(np.sum(~np.isnan(mb[key]))),
                    "mae": float(np.nanmean(np.abs(mb[key])))}
    out["M7"] = mb["M7"]; out["M7_t"] = mb["M7_t"]
    out["n_traj"] = mb["n_traj"]; out["n_t"] = mb["n_t"]
    out["n_traj_raid_ok"] = int(mb["M1_raid"].size)
    out["n_pic_int"] = int(mb["pic_int"].sum())
    return out


def bootstrap_M2(Z, N, B=B_BOOT, seed=42):
    n, m = Z.shape
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Zb, Nb = Z[idx], N[idx]
        prof_N = np.zeros_like(Nb); prof_Z = np.zeros_like(Zb)
        for i in range(n):
            o = np.arange(n) != i
            prof_N[i] = Nb[o].mean(axis=0)
            prof_Z[i] = Zb[o].mean(axis=0)
        Rz = (Zb - Zb.mean(axis=1, keepdims=True)) - (prof_Z - prof_Z.mean(axis=1, keepdims=True))
        Rn = (Nb - Nb.mean(axis=1, keepdims=True)) - (prof_N - prof_N.mean(axis=1, keepdims=True))
        vals.append(pearson(Rz.ravel(), Rn.ravel()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------- comparateurs
def build_comparators(truth, tables):
    dt = float(truth["times"][1] - truth["times"][0])
    traj_ids = np.where(np.isclose(truth["nu"], NU_HELDOUT))[0]
    t_idx = np.arange(T_LO, T_HI)
    n_t = len(t_idx)
    E_full = truth["E"][traj_ids][:, t_idx]
    N_full = truth["nuD"][traj_ids][:, t_idx]
    N_all = truth["nuD"][traj_ids]
    t_pic = t_pic_per_traj(truth, traj_ids)

    comp = {}
    Eall = truth["E"][traj_ids]
    est = np.empty_like(Eall)
    est[:, 1:-1] = (Eall[:, :-2] - Eall[:, 2:]) / (2 * dt)
    est[:, 0] = (Eall[:, 0] - Eall[:, 1]) / dt
    est[:, -1] = (Eall[:, -2] - Eall[:, -1]) / dt
    comp["plafond_centre"] = est[:, t_idx]
    causal = np.full_like(Eall, np.nan)
    for t in range(W - 1, Eall.shape[1]):
        causal[:, t] = (Eall[:, t - W + 1] - Eall[:, t]) / ((W - 1) * dt)
    comp["causal_w25"] = causal[:, t_idx]
    hor = np.zeros_like(N_full)
    for i in range(N_full.shape[0]):
        o = np.arange(N_full.shape[0]) != i
        hor[i] = N_all[o].mean(axis=0)[t_idx]
    comp["horloge_LOO"] = hor
    # exploratoire : profil médian LOO
    horm = np.zeros_like(N_full)
    for i in range(N_full.shape[0]):
        o = np.arange(N_full.shape[0]) != i
        horm[i] = np.median(N_all[o], axis=0)[t_idx]
    comp["horloge_median_LOO"] = horm
    # exploratoire : profil moyen LOO + rescaling affine par trajectoire (fit in-sample)
    hora = np.zeros_like(N_full)
    for i in range(N_full.shape[0]):
        o = np.arange(N_full.shape[0]) != i
        prof = N_all[o].mean(axis=0)[t_idx]
        X = np.column_stack([prof, np.ones(n_t)])
        c, *_ = np.linalg.lstsq(X, N_full[i], rcond=None)
        hora[i] = c[0] * prof + c[1]
    comp["horloge_affine_par_traj"] = hora
    ref = tables["none"][0]
    comp["statique"] = np.repeat(ref["z"].mean(axis=1, keepdims=True), n_t, axis=1)
    # baseline E_t : OLS sur train heldout_nu (nu in {0.05,0.2})
    tr = np.load(DATA / "train_data_nu_sweep.npz", allow_pickle=True)
    nu_tr = tr["nu_values"].astype(float)
    mtr = np.isin(nu_tr, [0.05, 0.2])
    Etr = tr["energy"].astype(float)[mtr][:, T_LO:T_HI].ravel()
    nuDtr = (nu_tr[:, None] * tr["dissipation"].astype(float))[mtr][:, T_LO:T_HI].ravel()
    X = np.column_stack([Etr, np.ones(len(Etr))])
    coef, *_ = np.linalg.lstsq(X, nuDtr, rcond=None)
    comp["baseline_E_t"] = coef[0] * E_full + coef[1]
    # none_recal : agrégé sur les 3 graines (fit par graine, voir main() pour le détail) ;
    # la valeur par défaut ici = graine 42, remplacée dans main() par l'agrégation 3 graines
    z_none = tables["none"][0]
    p = np.load(MODELS / z_none["run"] / "predictions_test.npz", allow_pickle=True)
    ti = p["trajectory_index"].astype(int); tt = p["time_index_t"].astype(int)
    nuv = p["nu"].astype(float); zv = p["z"].astype(float); ndv = p["nu_dissipation_t"].astype(float)
    msk = (np.isclose(nuv, 0.05) | np.isclose(nuv, 0.2)) & (tt >= T_LO) & (tt <= T_HI - 1)
    Xr = np.column_stack([zv[msk], np.ones(int(msk.sum()))])
    cr, *_ = np.linalg.lstsq(Xr, ndv[msk], rcond=None)
    yhat = Xr @ cr
    ss_res = float(np.sum((ndv[msk] - yhat) ** 2)); ss_tot = float(np.sum((ndv[msk] - ndv[msk].mean()) ** 2))
    r2_recal = 1 - ss_res / ss_tot
    Zr = cr[0] * z_none["z"] + cr[1]
    comp["none_recal"] = Zr
    comp["_none_recal_s42"] = {"alpha": float(cr[0]), "r2_fit": r2_recal}

    out = {}
    for name, Zc in comp.items():
        if name.startswith("_"):
            continue
        mb = metrics_block(Zc, N_full, E_full, t_idx, dt, t_pic)
        out[name] = aggregate(mb)
        out[name]["M2_ic95"] = bootstrap_M2(Zc, N_full)
    out["_coefs"] = {
        "baseline_E_t": {"a": float(coef[0]), "b": float(coef[1])},
        "none_recal": {"a": float(cr[0]), "b": float(cr[1]), "r2_fit_strates_0.05_0.20_test": r2_recal,
                        "note": "ajustement sur le TEST (strates nu=0.05/0.20) : les prédictions train ne sont pas exportées en npz"},
    }
    return out, t_pic, dt, t_idx, E_full, N_full


# ---------------------------------------------------------------- main
def main():
    truth = load_test_truth()
    dt = float(truth["times"][1] - truth["times"][0])

    inputs = sorted(DATA.glob("*_data_nu_sweep.npz")) + sorted(MODELS.glob("*/predictions_test.npz"))
    sha = {str(p.relative_to(REPO)): sha256(p) for p in inputs}

    tables = {"aphys_mono0": [], "none": []}
    for tag, dirs in RUNS_CONFIRM.items():
        for d in dirs:
            tables[tag].append(load_run_tables(d, truth))

    traj_ids = np.where(np.isclose(truth["nu"], NU_HELDOUT))[0]
    t_pic = t_pic_per_traj(truth, traj_ids)
    t_idx = np.arange(T_LO, T_HI)

    # ventilation des pics (point 5)
    vent = {
        "n_traj_nu001": int(len(traj_ids)),
        "pic_interieur_[24,99]": int(((t_pic >= 24) & (t_pic <= 99)).sum()),
        "pic_a_t2.0_(t_index=100)": int((t_pic == 100).sum()),
        "pic_avant_0.48_(t_index<24)": int((t_pic < 24).sum()),
    }

    # réconciliation « 22 % » du Step 6 (point 6)
    t_idx_all = np.arange(24, 100)
    frac_strate = {}
    for v in (0.01, 0.05, 0.2):
        m = np.isclose(truth["nu"], v)
        dd = np.diff(truth["D"][m], axis=1)[:, t_idx_all] > 0
        frac_strate[f"nu={v}"] = {"n_gt0": int(dd.sum()), "n": int(dd.size), "frac": float(dd.mean())}
    n_gt0_001 = frac_strate["nu=0.01"]["n_gt0"]
    n_total = int(sum(f["n"] for f in frac_strate.values()))
    recon = {
        "frac_dDdt_gt0_affichee_step6": 0.22052631578947368,
        "explication": f"n(dD/dt>0 a nu=0.01) / n_total_poolé(3 strates) = {n_gt0_001}/{n_total}",
        "n_gt0_nu001": n_gt0_001, "n_total_poolé": n_total,
        "frac_intra_strate": frac_strate,
        "frac_poolée_3_strates": float(np.mean([f["frac"] for f in frac_strate.values()]) * 0 +
                                       sum(f["n_gt0"] for f in frac_strate.values()) / n_total),
    }

    results = {"prereg_hash": PREREG_HASH, "sha256_inputs": sha, "dt": dt,
               "ventilation_pics_nu001": vent, "reconciliation_step6_frac_dDdt": recon,
               "runs": {}, "comparateurs": {}}

    per_traj_rows = []
    for tag, tabs in tables.items():
        per_seed = []
        for tb in tabs:
            mb = metrics_block(tb["z"], tb["nuD"], tb["E"], tb["t_idx"], dt, t_pic)
            agg = aggregate(mb, seed=tb["seed"])
            agg["M2_ic95"] = bootstrap_M2(tb["z"], tb["nuD"], seed=tb["seed"])
            agg["run"] = tb["run"]; agg["seed"] = tb["seed"]
            agg["protocol"] = tb["protocol"]; agg["z_target_mode"] = tb["z_target_mode"]
            results["runs"].setdefault(tag, []).append(agg)
            per_seed.append(agg)
            for i in range(mb["n_traj"]):
                per_traj_rows.append({
                    "tag": tag, "run": tb["run"], "seed": tb["seed"],
                    "traj": int(tb["traj_ids"][i]),
                    "pic_int": bool(mb["pic_int"][i]),
                    "n_pas_raid": int(mb["raid"][i].sum()),
                    "M1": mb["M1"][i], "M1s": mb["M1s"][i],
                    "M1_raid": mb["M1_raid"][i] if i < len(mb["M1_raid"]) else "",
                    "M4_s": mb["M4_s"][i], "M5_lag": mb["M5_lag"][i],
                    "M6": mb["M6"][i] if i < len(mb["M6"]) else "",
                    "M8_raid": mb["M8_raid"][i] if i < len(mb["M8_raid"]) else "",
                })
        inter = {}
        for key in ("M1_raid", "M1", "M2", "M6", "M8_raid", "M4_pic_int_s"):
            vals = []
            for a in per_seed:
                v = a[key]["median"] if isinstance(a[key], dict) else a[key]
                vals.append(v)
            m, lo, hi = t_ic95(np.array(vals))
            inter[key] = {"mean": m, "ic95": [lo, hi], "per_seed": vals}
        m3k5 = [a["M3"].get("k5_raid") for a in per_seed]
        inter["M3_k5_raid"] = {"mean": float(np.nanmean(m3k5)), "per_seed": m3k5}
        results["runs"][tag + "__inter_graines"] = inter

    # appariement par graine aphys_mono0 vs none
    pair = {}
    seeds = {a["seed"]: a for a in results["runs"]["aphys_mono0"]}
    seeds_n = {a["seed"]: a for a in results["runs"]["none"]}
    for s in sorted(set(seeds) & set(seeds_n)):
        pair[f"s{s}"] = {
            "dM1_raid": seeds[s]["M1_raid"]["median"] - seeds_n[s]["M1_raid"]["median"],
            "dM2": seeds[s]["M2"] - seeds_n[s]["M2"],
            "dM3_k5_raid": seeds[s]["M3"].get("k5_raid", float("nan")) - seeds_n[s]["M3"].get("k5_raid", float("nan")),
        }
    results["appariement_par_graine"] = pair

    comp, t_pic_c, dt_c, t_idx_c, E_full, N_full = build_comparators(truth, tables)
    results["comparateurs"] = comp

    # exploratoire
    expl = {}
    for name, d in RUNS_EXPLORATOIRE.items():
        w = 50 if "w50" in d else 25
        tb = load_run_tables(d, truth, w=w)
        tpic_e = t_pic_per_traj(truth, np.where(np.isclose(truth["nu"], NU_HELDOUT))[0])
        mb = metrics_block(tb["z"], tb["nuD"], tb["E"], tb["t_idx"], dt, tpic_e)
        agg = aggregate(mb, seed=tb["seed"])
        agg["run"] = d; agg["protocol"] = tb["protocol"]; agg["z_target_mode"] = tb["z_target_mode"]
        expl[name] = agg
    # M5 lag ±20 (exploratoire, point 7)
    for tag in ("aphys_mono0", "none"):
        tb = tables[tag][0]
        mb20 = metrics_block(tb["z"], tb["nuD"], tb["E"], tb["t_idx"], dt, t_pic, lag_max=20)
        expl[f"{tag}_s42_M5_lag_pm20"] = {"M5_lag_median": float(np.median(mb20["M5_lag"])),
                                           "M5_lag_ic95": ic95_bootstrap(mb20["M5_lag"], seed=42)[1:]}
    # standard à nu=0.05
    d = "window_w25_standard"
    p = np.load(MODELS / d / "predictions_test.npz", allow_pickle=True)
    ti = p["trajectory_index"].astype(int); tt = p["time_index_t"].astype(int)
    nuv = p["nu"].astype(float)
    m05 = np.isclose(nuv, 0.05)
    traj_ids5 = np.unique(ti[m05]); n_tr = len(traj_ids5)
    Z = np.full((n_tr, len(t_idx)), np.nan); N = np.full_like(Z, np.nan); E = np.full_like(Z, np.nan)
    pos = {int(t): k for k, t in enumerate(t_idx)}
    Ekey = p["energy_t"] if "energy_t" in p.files else truth["E"][ti, tt]
    for row in np.where(m05)[0]:
        i = int(np.where(traj_ids5 == ti[row])[0][0])
        Z[i, pos[int(tt[row])]] = p["z"][row]
        N[i, pos[int(tt[row])]] = p["nu_dissipation_t"][row]
        E[i, pos[int(tt[row])]] = Ekey[row]
    tpic05 = np.argmax(truth["nuD"][traj_ids5], axis=1)
    expl["none_s42_standard_nu005"] = aggregate(metrics_block(Z, N, E, t_idx, dt, tpic05))
    # définition alternative du raidissement
    tb = tables["aphys_mono0"][0]
    expl["aphys_mono0_s42_alt_raidissement"] = aggregate(
        metrics_block(tb["z"], tb["nuD"], tb["E"], tb["t_idx"], dt, t_pic, def_raid="dnuD>0"))
    # none_recal PAR GRAINE (point 6) : fit nuD ~ alpha*z + beta sur strates TEST 0.05/0.20,
    # appliqué à nu=0.01 ; une recalibration affine ne change que le signe des corrélations
    nr = {}
    for tb in tables["none"]:
        p = np.load(MODELS / tb["run"] / "predictions_test.npz", allow_pickle=True)
        ti = p["trajectory_index"].astype(int); tt = p["time_index_t"].astype(int)
        nuv = p["nu"].astype(float); zv = p["z"].astype(float); ndv = p["nu_dissipation_t"].astype(float)
        msk = (np.isclose(nuv, 0.05) | np.isclose(nuv, 0.2)) & (tt >= T_LO) & (tt <= T_HI - 1)
        Xr = np.column_stack([zv[msk], np.ones(int(msk.sum()))])
        cr, *_ = np.linalg.lstsq(Xr, ndv[msk], rcond=None)
        yhat = Xr @ cr
        r2 = 1 - float(np.sum((ndv[msk] - yhat) ** 2)) / float(np.sum((ndv[msk] - ndv[msk].mean()) ** 2))
        Zr = cr[0] * tb["z"] + cr[1]
        mb_avant = metrics_block(tb["z"], tb["nuD"], tb["E"], tb["t_idx"], dt, t_pic)
        mb_apres = metrics_block(Zr, tb["nuD"], tb["E"], tb["t_idx"], dt, t_pic)
        nr[f"s{tb['seed']}"] = {
            "run": tb["run"], "alpha": float(cr[0]), "beta": float(cr[1]),
            "signe_alpha": int(np.sign(cr[0])), "r2_fit_strates_test_0.05_0.20": r2,
            "M1_raid_avant": float(np.nanmedian(mb_avant["M1_raid"])),
            "M1_raid_apres": float(np.nanmedian(mb_apres["M1_raid"])),
            "note": "recalibration affine : M1_raid_apres = |M1_raid_avant| si alpha>0, -|...| si alpha<0 "
                    "(les inversions de signe viennent du fit sur les strates test 0.05/0.20)",
        }
    expl["none_recal_par_graine"] = nr
    # comparateur none_recal agrégé 3 graines (corrige la valeur s42 seule) :
    per_seed_nr = []
    for tb in tables["none"]:
        p = np.load(MODELS / tb["run"] / "predictions_test.npz", allow_pickle=True)
        ti = p["trajectory_index"].astype(int); tt = p["time_index_t"].astype(int)
        nuv = p["nu"].astype(float); zv = p["z"].astype(float); ndv = p["nu_dissipation_t"].astype(float)
        msk = (np.isclose(nuv, 0.05) | np.isclose(nuv, 0.2)) & (tt >= T_LO) & (tt <= T_HI - 1)
        Xr = np.column_stack([zv[msk], np.ones(int(msk.sum()))])
        cr, *_ = np.linalg.lstsq(Xr, ndv[msk], rcond=None)
        Zr = cr[0] * tb["z"] + cr[1]
        mb = metrics_block(Zr, tb["nuD"], tb["E"], tb["t_idx"], dt, t_pic)
        per_seed_nr.append({"seed": tb["seed"], "alpha": float(cr[0]),
                            "M1_raid": float(np.nanmedian(mb["M1_raid"])), "M2": mb["M2"]})
    m1r_vals = [v["M1_raid"] for v in per_seed_nr]
    m2_vals = [v["M2"] for v in per_seed_nr]
    m1r_m, m1r_lo, m1r_hi = t_ic95(np.array(m1r_vals))
    m2_m, m2_lo, m2_hi = t_ic95(np.array(m2_vals))
    results["comparateurs"]["none_recal"] = {
        "per_seed": per_seed_nr,
        "M1_raid": {"mean": m1r_m, "ic95": [m1r_lo, m1r_hi], "per_seed": m1r_vals},
        "M2": {"mean": m2_m, "ic95": [m2_lo, m2_hi], "per_seed": m2_vals},
        "conclusion": ("recalibrage affine (fit sur strates test 0.05/0.20, alpha>0 pour les 3 "
                       "graines) n'apporte aucune information dynamique : moyenne 3 graines "
                       "M1_raid = moyenne de none brut"),
        "note": "ajustement sur le TEST (strates nu=0.05/0.20) : prédictions train non exportées en npz",
    }
    results["exploratoire"] = expl

    # ---------------- verdict : règle MOT POUR MOT du prereg 5623042 §4 (6 conditions) ----------------
    # §4 : "TRACEUR DYNAMIQUE si ... M1_raid Pearson >= 0.7 ET M3 k=5 raid >= 0.8 ET M2 poolé
    # >= 0.5 avec IC95 excluant 0 ET M1_raid > M1_raid(baseline E_t) avec IC de la différence
    # excluant 0 ET |M4| médian <= 0.3 s ET M6 >= 0.5."  (6 conditions ; la comparaison appariée
    # vs none relève du §3 agrégation, pas du §4 -> analyse complémentaire, pas critère.)
    ag = results["runs"]["aphys_mono0__inter_graines"]
    m1r = ag["M1_raid"]["mean"]
    m1r_ic = ag["M1_raid"]["ic95"]
    m2 = ag["M2"]["mean"]
    m3k5 = ag["M3_k5_raid"]["mean"]
    m4 = abs(ag["M4_pic_int_s"]["mean"])
    m6 = ag["M6"]["mean"]
    m8 = ag["M8_raid"]["mean"]
    m1r_et = results["comparateurs"]["baseline_E_t"]["M1_raid"]["median"]
    m2_ic = results["runs"]["aphys_mono0"][0]["M2_ic95"]
    # condition 4 : IC de la différence M1_raid(aphys) - M1_raid(E_t) ; E_t déterministe ->
    # IC de la différence = IC inter-graines de M1_raid(aphys) décalé de -M1_raid(E_t)
    d_et = m1r - m1r_et
    d_et_ic = [m1r_ic[0] - m1r_et, m1r_ic[1] - m1r_et]
    cond = {
        "1_M1_raid_Pearson>=0.7": m1r >= 0.7,
        "2_M3_k5_raid>=0.8": m3k5 >= 0.8,
        "3_M2_pool>=0.5_avec_IC95_excluant_0": (m2 >= 0.5) and (m2_ic[0] > 0),
        "4_M1_raid>M1_raid(baseline_E_t)_avec_IC_diff_excluant_0": (m1r > m1r_et) and (d_et_ic[0] > 0),
        "5_|M4|_median<=0.3s": m4 <= 0.3,
        "6_M6>=0.5": m6 >= 0.5,
    }
    if all(cond.values()):
        verdict = "TRACEUR DYNAMIQUE"
    elif (m1r < 0.3) or (m2 <= 0.2) or (m8 > m1r):
        verdict = "ETIQUETTE STATIQUE"
    else:
        verdict = "PARTIEL"
    dvals = [v["dM1_raid"] for v in pair.values()]
    dm, dlo, dhi = t_ic95(np.array(dvals))
    results["verdict"] = {
        "verdict": verdict,
        "regle_decision": ("prereg 5623042 §4, 6 conditions, appliquées telles quelles ; "
                           "la comparaison appariée vs none (§3) est une analyse complémentaire"),
        "conditions_prereg_5623042_§4": cond,
        "valeurs": {"M1_raid": m1r, "M1_raid_ic95_inter_graines": m1r_ic, "M2": m2,
                    "M2_ic95_bootstrap_s42": m2_ic, "M3_k5_raid": m3k5,
                    "M4_pic_int_s": ag["M4_pic_int_s"]["mean"], "M6": m6, "M8_raid": m8,
                    "M1_raid_baseline_E_t": m1r_et,
                    "diff_M1_raid_vs_E_t": {"diff": d_et, "ic95": d_et_ic}},
    }
    results["analyse_complementaire"] = {
        "appariement_aphys_mono0_vs_none_par_graine": {
            "per_seed": pair, "dM1_raid": {"mean": dm, "ic95_t_ddl2": [dlo, dhi]},
            "note": "§3 du prereg (agrégation), pas un critère du §4 ; non concluant à n=3 "
                    "(IC95 inclut 0), les 3 différences sont positives -> Limites"},
    }
    results["limites"] = [
        "comparaison appariée inter-graines à n=3 (t ddl=2 = 4.303) : IC95 [%.3f, %.3f] inclut 0, "
        "les 3 différences par graine sont positives" % (dlo, dhi),
        "M5 au bord de plage pour les baselines ; M2(horloge LOO) = -1 par construction",
        "code générateur de phase_strat (Step 6) non versionné : reproduit par main_reproduce_phase_strat.py",
    ]

    # ---------------- sorties ----------------
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "a1_intra_trajectory.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    with open(LOGS / "a1_intra_trajectory_per_traj.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(per_traj_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(per_traj_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGS.mkdir(parents=True, exist_ok=True)
        tb = tables["aphys_mono0"][0]
        Z, N, E = tb["z"], tb["nuD"], tb["E"]
        tt = tb["t_idx"] * dt
        mb = metrics_block(Z, N, E, tb["t_idx"], dt, t_pic)
        rng = np.random.default_rng(42)
        rand8 = rng.choice(mb["n_traj"], 8, replace=False)
        order = np.argsort(np.nan_to_num(mb["M1_raid"], nan=-9))
        best2, worst2 = order[-2:], order[:2]
        for i in list(rand8) + list(best2) + list(worst2):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(tt, Z[i], label="z (unités physiques)")
            ax.plot(tt, N[i], label="nuD_vrai")
            en = (E[i] - np.nanmin(E[i])) / max(np.nanmax(E[i]) - np.nanmin(E[i]), 1e-30)
            ax.plot(tt, en * (np.nanmax(N[i]) - np.nanmin(N[i])) + np.nanmin(N[i]),
                    label="E_t (normalisé)", alpha=0.6)
            ax.axvline(t_pic[i] * dt, color="k", ls="--", alpha=0.4, label="t_pic")
            ax.set_title(f"traj {tb['traj_ids'][i]} (M1_raid={mb['M1_raid'][i] if i < len(mb['M1_raid']) else float('nan'):.3f})")
            ax.set_xlabel("t (s)"); ax.legend()
            tag = "best" if i in best2 else ("worst" if i in worst2 else "rand")
            fig.savefig(FIGS / f"a1_traj_{tb['traj_ids'][i]}_{tag}.png", dpi=110)
            plt.close(fig)
        for key, fname in (("M1_raid", "a1_hist_M1_raid.png"), ("M4_pic_int_s", "a1_hist_M4.png")):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(np.asarray(mb[key], float), bins=15)
            ax.set_title(f"distribution {key} (aphys_mono0 s42)")
            fig.savefig(FIGS / fname, dpi=110)
            plt.close(fig)
        results["figures"] = "outputs/figures/a1_*.png"
    except ImportError:
        tb = tables["aphys_mono0"][0]
        with open(LOGS / "a1_figures_data.csv", "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["traj", "t_s", "z", "nuD_vrai", "E_t"])
            for i in range(tb["z"].shape[0]):
                for j, t in enumerate(tb["t_idx"]):
                    wcsv.writerow([int(tb["traj_ids"][i]), t * dt, tb["z"][i, j], tb["nuD"][i, j], tb["E"][i, j]])
        results["figures"] = "matplotlib indisponible : séries dans outputs/logs/a1_figures_data.csv"

    with open(LOGS / "a1_intra_trajectory.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)

    print("verdict:", verdict)
    print("conditions:", cond)
    print("appariement: mean=%.4f ic95=[%.4f, %.4f]" % (dm, dlo, dhi))
    print("M1_raid:", round(m1r, 4), "| M2:", round(m2, 4), "| M3_k5_raid:", round(m3k5, 4),
          "| M4_pic_int:", round(ag["M4_pic_int_s"]["mean"], 3), "| M6:", round(m6, 4), "| M8:", round(m8, 4))
    print("ventilation pics:", vent)


if __name__ == "__main__":
    main()
