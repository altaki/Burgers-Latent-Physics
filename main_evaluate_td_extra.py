#!/usr/bin/env python3
"""Évaluateur ADDITIF — strate test_extra ν=0.02 (prereg manuscript/prereg_nu002_strate_test.{md,json}).

INFERENCE SEULE : lit les checkpoints existants (hash SHA-256 avant/après), aucun
réentraînement, aucun `.lock`, aucune écriture dans `outputs/models/`. Les prédictions
par fenêtre sont exportées (additif) dans `--preds-dir` ; les résultats dans `--out`
(refus si le fichier existe).

Lit le prereg JSON et ÉCHOUE (exit 1) si l'ensemble des conditions évaluées diffère
de `conditions_evaluees_attendues`. Exit 1 sur toute valeur NaN des métriques.

Usage (env torch requis, Windows/PowerShell) :
  python3 main_evaluate_td_extra.py --stratum outputs/data/test_extra_data_nu002.npz
  python3 main_evaluate_td_extra.py --smoke     # logique pure, sans torch, écrit dans /tmp uniquement
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import main_evaluate_td as ev  # numpy-only : spearman, m6_raidissement, run_metrics réutilisés
from main_export_predictions_val import build_td_model_from_config  # dispatch encoder_mode (td0 vs td1/2/3)

DATA_DIR = REPO / "outputs" / "data"
MODELS_ROOT = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"
TRAIN_NPZ = DATA_DIR / "train_data_nu_sweep.npz"
PREREG_JSON = REPO / "manuscript" / "prereg_nu002_strate_test.json"
WINDOW_SIZE = 25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- fonctions pures (testables sans torch) ----------

def r2_identity(z: np.ndarray, y: np.ndarray) -> float:
    z = np.asarray(z, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(np.sum((z - y) ** 2)) / ss_tot


def fit_affine_calibration(z_fit: np.ndarray, y_fit: np.ndarray):
    """OLS y ~ a*z + b (catégorie 3 : fit hors strate, évalué ailleurs)."""
    A = np.vstack([np.asarray(z_fit, float).reshape(-1), np.ones(len(z_fit))]).T
    coef, _, _, _ = np.linalg.lstsq(A, np.asarray(y_fit, float).reshape(-1), rcond=None)
    return float(coef[0]), float(coef[1])


def unil_bounds(d: np.ndarray, t95: float = 2.1318):
    """IC unilatéral 95 % de la moyenne de d (n=5, t ddl=4)."""
    d = np.asarray(d, float).reshape(-1)
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1))
    half = t95 * s / np.sqrt(len(d))
    return m, s, m - half, m + half  # mean, sd, lower, upper


def verdict_etat(L: float, U: float, delta_R: float) -> str:
    """4 états mutuellement exclusifs E1-E4 (prereg JSON decision_rule.etats)."""
    if L > 0 and U < delta_R:
        return "E1_POSITIF_BORNE"
    if L > 0 and U >= delta_R:
        return "E2_POSITIF_NON_BORNE"
    if L <= 0 and U < delta_R:
        return "E3_NEGATIF"
    return "E4_INDETERMINE"


def check_conditions(found: set, expected: set) -> None:
    if found != expected:
        raise SystemExit(
            f"[ABORT exit 1] conditions évaluées != prereg JSON : "
            f"manquantes={sorted(expected - found)} en_trop={sorted(found - expected)}"
        )


def nan_guard(metrics: dict, where: str) -> None:
    bad = [k for k, v in metrics.items() if isinstance(v, float) and np.isnan(v)]
    if bad:
        raise SystemExit(f"[ABORT exit 1] NaN détecté ({where}) dans : {bad}")


def train_nuD_range_per_protocole(train_npz: Path) -> dict:
    """max νD sur les fenêtres d'entraînement (t ∈ [24, 99]) par protocole, numpy pur."""
    d = np.load(train_npz, allow_pickle=True)
    nu = np.asarray(d["nu_values"], float)
    diss_phys = np.asarray(d["dissipation_physical"], float)  # (n_traj, n_times)
    out = {}
    for proto, allowed in (("heldout_nu", (0.05, 0.2)), ("heldout_nu005", (0.01, 0.2))):
        m = np.any(np.isclose(nu[:, None], np.asarray(allowed)[None, :]), axis=1)
        out[proto] = float(np.max(diss_phys[np.ix_(m, np.arange(24, 100))]))
    out["train_complet"] = float(np.max(diss_phys[:, 24:100]))
    return out


# ---------- inférence (torch, côté Windows uniquement) ----------

def predict_run_on_stratum(run_dir: Path, extra_npz: Path, device: str = "cpu") -> dict:
    import torch
    import ml_train_experiment_a_seq as mt

    ckpt_path = run_dir / "best_model.pt"
    sha_before = sha256_file(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    datasets_module = mt._load_datasets_module()
    ec_datasets = mt._load_experiment_c_datasets_module()
    train_raw = datasets_module.load_dataset_npz(TRAIN_NPZ)  # standardiseur fit sur train nu_sweep, comme à l'entraînement
    standardizer = datasets_module.fit_global_standardizer(np.asarray(train_raw["states"], dtype=np.float32))
    extra_raw = datasets_module.load_dataset_npz(extra_npz)  # lecture seule
    ds = ec_datasets.NuSweepWindowPairDataset(
        extra_raw, window_size=WINDOW_SIZE, stride=1, normalize=True,
        standardizer=standardizer, nu_filter=None, return_metadata=True,
    )
    ec_models = mt._load_experiment_c_models_module()
    model = build_td_model_from_config(ec_models, ckpt.get("model_config"), dt=float(ds.dt))
    model.load_state_dict(ckpt["model_state_dict"])
    pred = mt.predict_a_seq(model, ds, device=torch.device(device))
    z = pred["z"]
    zmode = run_dir.name.split("_z")[-1] if "_z" in run_dir.name else "none"
    std_path = run_dir / "standardizers.json"
    if std_path.exists() and zmode != "none":
        std = json.loads(std_path.read_text(encoding="utf-8")).get("z_target_standardizer")
        if std:
            z = z * float(std["std"]) + float(std["mean"])
    return {
        "z": z, "h": pred["h"], "meta": pred["meta"],
        "checkpoint_sha256_before": sha_before,
        "checkpoint_sha256_after": sha256_file(ckpt_path),
    }


# ---------- évaluation principale ----------

def evaluate(stratum_npz: Path, prereg_json: Path, out: Path, preds_dir: Path, val_dir: Path, device: str = "cpu") -> int:
    prereg = json.loads(Path(prereg_json).read_text(encoding="utf-8"))
    expected = set(prereg["conditions_evaluees_attendues"]["runs"])
    if out.exists():
        print(f"[ABORT] {out} existe déjà (aucune écrasement).")
        return 1
    nu = float(prereg["stratum"]["nu"])
    delta_R = float(prereg["decision_rule"]["delta_R"])
    t95 = float(prereg["decision_rule"]["t_0.95_4"])

    train_ranges = train_nuD_range_per_protocole(TRAIN_NPZ)

    per_run = {}
    found = set()
    for run_name in sorted(expected):
        run_dir = MODELS_ROOT / run_name
        if not (run_dir / "best_model.pt").exists():
            continue  # sera signalé par check_conditions
        found.add(run_name)
        pred = predict_run_on_stratum(run_dir, stratum_npz, device)
        meta = pred["meta"]
        y = np.asarray(meta["nu_dissipation_t"], float)
        z = np.asarray(pred["z"], float)
        # métriques brutes (co-primaire i)
        m = {
            "r2_identity": r2_identity(z, y),
            "spearman": ev.spearman(z, y),
            "calibration_slope": float(np.polyfit(y, z, 1)[0]) if np.std(z) > 0 else float("nan"),
            "rel_rmse": float(np.sqrt(np.mean((z - y) ** 2)) / np.mean(np.abs(y))),
            "n_windows": int(len(z)),
        }
        # co-primaire (ii) : calibration fit sur les prédictions de VALIDATION du run
        val_npz = val_dir / f"{run_name}_predictions_val.npz"
        if not val_npz.exists():
            raise SystemExit(f"[ABORT exit 1] export val manquant : {val_npz} (lancer main_export_predictions_val.py d'abord)")
        dv = np.load(val_npz, allow_pickle=True)
        a, b = fit_affine_calibration(dv["z"], dv["nu_dissipation_t"])
        m["calib_a"], m["calib_b"] = a, b
        m["r2_calibree_hors_strate"] = r2_identity(a * z + b, y)
        # M6 (définition prereg A1 5623042, via ev.m6_raidissement sur npz-like dict)
        nuD_traj = {}
        for ti in np.unique(meta["trajectory_index"]):
            sel = meta["trajectory_index"] == ti
            nuD_traj[int(ti)] = np.asarray(meta["nu_dissipation_t"])[sel]
        d_npz = {"nu": np.full(len(z), nu), "z": z, "D": np.asarray(meta["dissipation_t"], float),
                 "h": pred["h"], "z_target_mode": np.array(run_dir.name.split("_z")[-1] if "_z" in run_dir.name else "none")}
        try:
            m["M6_raidissement"] = ev.m6_raidissement(d_npz, np.ones(len(z), bool), nu, nuD_traj)
        except Exception:
            m["M6_raidissement"] = float("nan")
        m["checkpoint_unchanged"] = pred["checkpoint_sha256_before"] == pred["checkpoint_sha256_after"]
        # export additif des prédictions extra (jamais dans le run dir)
        preds_dir.mkdir(parents=True, exist_ok=True)
        ppath = preds_dir / f"{run_name}_predictions_extra_nu002.npz"
        if ppath.exists():
            raise SystemExit(f"[ABORT exit 1] {ppath} existe déjà")
        import ml_train_experiment_a_seq as mt
        mt.save_predictions_npz(
            ppath, z=z, h=pred["h"], meta=meta, u=pred.get("u"), dx=float(extra_raw_dx(stratum_npz)),
            split="test_extra", protocol=str(np.load(stratum_npz, allow_pickle=True)["protocol"]),
            seed=int(run_dir.name.rsplit("_s", 1)[1]), window_size=WINDOW_SIZE,
            z_target_mode=str(d_npz["z_target_mode"]), run_name=run_name,
            extra={"source": np.array("td_extra_eval"), "stratum": np.array(str(stratum_npz)),
                   "checkpoint_sha256": np.array(pred["checkpoint_sha256_before"])},
        )
        nan_guard(m, run_name)
        per_run[run_name] = m
        print(f"[eval] {run_name}: r2_id={m['r2_identity']:.4f} r2_cal={m['r2_calibree_hors_strate']:.4f} ckpt_ok={m['checkpoint_unchanged']}")

    check_conditions(found, expected)  # exit 1 si différence

    # comparaisons appariées par graine (mode z uniquement), par protocole
    comp_defs = prereg["comparaisons"]
    bras = prereg["bras"]
    seeds = prereg["seeds"]
    results_comparisons = {}
    for proto in prereg["protocoles"]:
        pid = proto["id"]
        for comp in comp_defs:
            if comp.get("protocole") not in (None, pid) or comp["statut"] == "NON_CALCULABLE":
                continue
            hi, lo = comp["bras_haut"], comp["bras_bas"]
            d_id, d_cal = [], []
            for s in seeds:
                rhi = per_run[f"{hi}_{pid}_zenergy_centered_s{s}"]
                rlo = per_run[f"{lo}_{pid}_zenergy_centered_s{s}"]
                d_id.append(rhi["r2_identity"] - rlo["r2_identity"])
                d_cal.append(rhi["r2_calibree_hors_strate"] - rlo["r2_calibree_hors_strate"])
            m1, s1, l1, u1 = unil_bounds(d_id, t95)
            m2, s2, l2, u2 = unil_bounds(d_cal, t95)
            L, U = min(l1, l2), max(u1, u2)
            etat = verdict_etat(L, U, delta_R)
            nan_guard({"mean_id": m1, "mean_cal": m2}, f"{comp['id']}/{pid}")
            results_comparisons[f"{comp['id']}|{pid}"] = {
                "statut": comp["statut"], "mean_r2_id": m1, "sd_r2_id": s1,
                "mean_r2_cal": m2, "sd_r2_cal": s2, "L": L, "U": U,
                "verdict": etat, "delta_R": delta_R,
            }
            print(f"[verdict] {comp['id']} @ {pid} ({comp['statut']}): mean_id={m1:+.4f} mean_cal={m2:+.4f} L={L:+.4f} U={U:+.4f} -> {etat}")

    # plages vs prédictions (moyenne des 5 graines, par bras, mode z)
    plages = {}
    for pid, pred_block in prereg["predictions_nu002"].items():
        if pid not in ("heldout_nu", "heldout_nu005"):
            continue
        for bras_name, spec in pred_block["r2_par_bras"].items():
            vals = [per_run[f"{bras_name}_{pid}_zenergy_centered_s{s}"]["r2_identity"] for s in seeds]
            mv = float(np.mean(vals))
            lo, hi = spec["plage_moy_moins_2sd"]
            plages[f"{bras_name}|{pid}"] = {"moy_observee": mv, "plage": [lo, hi],
                                             "dans_plage": bool(lo <= mv <= hi)}
    # stratification in/hors plage νD
    d_extra = np.load(stratum_npz, allow_pickle=True)
    strat = {}
    for pid, nuDmax in train_ranges.items():
        if pid == "train_complet":
            continue
        strat[pid] = {"nuD_train_max": nuDmax}
    results = {
        "experiment": prereg["experiment"], "stratum_npz": str(stratum_npz),
        "stratum_sha256": sha256_file(stratum_npz), "nu": nu,
        "train_nuD_ranges": train_ranges, "per_run": per_run,
        "comparisons": results_comparisons, "plages": plages, "stratification": strat,
        "n_runs_evalues": len(found),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[eval] résultats -> {out} ({len(found)} runs)")
    return 0


def extra_raw_dx(stratum_npz: Path) -> float:
    d = np.load(stratum_npz, allow_pickle=True)
    return float(d["dx"])


# ---------- smoke (sans torch, aucune écriture dans outputs/) ----------

def smoke(smoke_dir: Path) -> int:
    smoke_dir.mkdir(parents=True, exist_ok=True)
    rep = {"smoke": True, "checks": {}}
    # (a) lecture seule d'une strate existante (clés/formes)
    d = np.load(DATA_DIR / "test_data_nu_sweep.npz", allow_pickle=True)
    rep["checks"]["strates_existantes_lues"] = bool(d["states"].shape == (150, 101, 1024) and "trajectory_seeds" in d.files)
    # (b) métriques + verdicts sur données synthétiques
    rng = np.random.default_rng(0)
    y = rng.normal(0.2, 0.1, 11400)
    z = 0.9 * y + 0.02 + rng.normal(0, 0.01, 11400)
    rep["checks"]["r2_identity_synth"] = round(r2_identity(z, y), 4)
    a, b = fit_affine_calibration(z[:5000], y[:5000])
    rep["checks"]["calib_synth"] = [round(a, 4), round(b, 4)]
    # E1 attendu : L>0, U<0.05
    d_id = np.array([0.027, 0.028, 0.026, 0.029, 0.027])
    d_cal = np.array([0.021, 0.022, 0.020, 0.023, 0.021])
    m1, s1, l1, u1 = unil_bounds(d_id)
    m2, s2, l2, u2 = unil_bounds(d_cal)
    etat = verdict_etat(min(l1, l2), max(u1, u2), 0.05)
    rep["checks"]["verdict_E1_attendu"] = etat
    rep["checks"]["E1_ok"] = etat == "E1_POSITIF_BORNE"
    # E2 / E3 / E4
    rep["checks"]["E2"] = verdict_etat(0.06, 0.09, 0.05)
    rep["checks"]["E3"] = verdict_etat(-0.01, 0.04, 0.05)
    rep["checks"]["E4"] = verdict_etat(-0.01, 0.06, 0.05)
    # (c) conditions : mismatch -> SystemExit
    try:
        check_conditions({"a"}, {"a", "b"})
        rep["checks"]["conditions_mismatch_non_detecte"] = True
    except SystemExit:
        rep["checks"]["conditions_mismatch_exit1_ok"] = True
    # (d) NaN -> exit 1
    try:
        nan_guard({"r2": float("nan")}, "smoke")
        rep["checks"]["nan_non_detecte"] = True
    except SystemExit:
        rep["checks"]["nan_exit1_ok"] = True
    # (e) plage train νD (numpy pur, lecture seule)
    rep["train_nuD_ranges"] = train_nuD_range_per_protocole(TRAIN_NPZ)
    # README:28 : max νD train (protocole heldout_nu, ν=0.01 exclu) = 1.8121
    rep["heldout_nu_max_nuD_attendu_1.8121_readme28"] = abs(rep["train_nuD_ranges"]["heldout_nu"] - 1.8121) < 1e-3
    # (f) dispatch du modèle depuis model_config (sans torch, faux module)
    class _FakeModels:
        def __init__(self):
            self.calls = []
        def build_window_dynamics_autoencoder(self, **kw):
            self.calls.append(("td0_builder", sorted(kw)))
            return "m_td0"
        def build_window_dynamics_autoencoder_td(self, encoder_mode, dt, **kw):
            # dt en paramètre nommé, comme le vrai builder (ml_models_experiment_c.py:731)
            self.calls.append((f"td_builder[{encoder_mode}] dt={dt}", sorted(kw)))
            return f"m_{encoder_mode}"
    fake = _FakeModels()
    base_cfg = {"model": "window_dynamics_autoencoder", "loss": {"z_weight": 1.0},
                "encoder_mode": "td1", "window_size": 25, "nx": 1024, "latent_dim": 8,
                "conv_channels": [16, 32], "activation": "relu", "dropout": 0.0}
    m1 = build_td_model_from_config(fake, base_cfg, dt=1e-4)
    m2 = build_td_model_from_config(fake, {**base_cfg, "encoder_mode": "td0"}, dt=1e-4)
    m3 = build_td_model_from_config(fake, {k: v for k, v in base_cfg.items() if k != "encoder_mode"}, dt=1e-4)
    rep["checks"]["dispatch_td1_vers_builder_td_avec_dt"] = m1 == "m_td1" and fake.calls[0][0] == "td_builder[td1] dt=0.0001" and "dt" not in fake.calls[0][1]
    rep["checks"]["dispatch_td0_vers_builder_classique"] = m2 == "m_td0" and fake.calls[1][0] == "td0_builder"
    rep["checks"]["dispatch_sans_encoder_mode_default_td0"] = m3 == "m_td0"
    rep["checks"]["cles_model_loss_retirees"] = all("model" not in kw and "loss" not in kw for _, kw in fake.calls)
    out = smoke_dir / "eval_nu002_smoke_report.json"
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    ok = all(v is True for k, v in rep["checks"].items() if k.endswith("_ok")) and rep["checks"].get("E1_ok")
    print(f"[SMOKE] rapport -> {out} ; checks: {rep['checks']} ; train_nuD_ranges: {rep['train_nuD_ranges']}")
    print(f"[SMOKE] outputs/ non touché (écriture limitée à {smoke_dir})")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stratum", type=Path, default=DATA_DIR / "test_extra_data_nu002.npz")
    ap.add_argument("--prereg", type=Path, default=PREREG_JSON)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "logs" / "nu002_eval_results.json")
    ap.add_argument("--preds-dir", type=Path, default=REPO / "outputs" / "predictions_extra_nu002")
    ap.add_argument("--val-dir", type=Path, default=REPO / "outputs" / "predictions_val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir", type=Path, default=Path("/tmp/nu002_eval_smoke"))
    args = ap.parse_args()
    if args.smoke:
        return smoke(args.smoke_dir)
    return evaluate(args.stratum, args.prereg, args.out, args.preds_dir, args.val_dir, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
