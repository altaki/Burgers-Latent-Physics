#!/usr/bin/env python3
"""Point d'entrée ADDITIF — génération de la strate test_extra à ν choisi (défaut 0.02).

Pré-enregistrement : `manuscript/prereg_nu002_strate_test.md` (commité AVANT usage).

Règles de sécurité (inchangées) :
- ne touche JAMAIS aux npz existants de `outputs/data/` (lecture seule,
  `allow_pickle=True`) : écrit UNIQUEMENT `--out` (refus si le fichier existe) et
  `--report` ;
- `main_generate_data.py` n'est PAS utilisé ni modifié ;
- `--smoke` écrit UNIQUEMENT dans `--smoke-dir` (défaut hors `outputs/`, ex. /tmp) ;
- seeds disjoints des 900 existants (vérifié contre les npz nu_sweep existants).

Usage (python3, racine du repo) :
  python3 main_generate_data_nu_extra.py --smoke                       # 2 traj -> /tmp, rien dans outputs/
  python3 main_generate_data_nu_extra.py --pilot 20 --n 0 --report outputs/logs/nu002_pilot_report.json
  python3 main_generate_data_nu_extra.py --pilot 0 --n 150             # génération + audit + npz + rapport
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

import pde_dataset_builder as b  # noqa: E402
from config import get_default_config  # noqa: E402

DATA_DIR = REPO / "outputs" / "data"
EXISTING_NU_SWEEP_NPZ = [
    DATA_DIR / "train_data_nu_sweep.npz",
    DATA_DIR / "val_data_nu_sweep.npz",
    DATA_DIR / "test_data_nu_sweep.npz",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_trajectory_seeds() -> np.ndarray:
    """Seeds des 900 trajectoires nu_sweep existantes (lecture seule)."""
    seeds = []
    for p in EXISTING_NU_SWEEP_NPZ:
        if not p.exists():
            raise FileNotFoundError(f"npz nu_sweep attendu introuvable (lecture seule) : {p}")
        d = np.load(p, allow_pickle=True)
        seeds.append(np.asarray(d["trajectory_seeds"], dtype=np.int64))
    return np.concatenate(seeds)


def nu_slug(nu: float) -> str:
    return f"nu{str(nu).replace('.', 'p')}"


def build_dataset(cfg, nu: float, n: int, rng: np.random.Generator) -> dict:
    ns = cfg.nu_sweep
    x = np.linspace(0.0, float(ns.L), int(ns.Nx), endpoint=False)
    dataset = b.generate_nu_sweep_split(
        n_trajectories=int(n),
        nu_values=[float(nu)],
        x=x,
        L=float(ns.L),
        T=float(ns.T),
        dt=float(ns.dt),
        save_every=int(ns.save_every),
        time_scheme=str(ns.time_scheme),
        ic_types=list(cfg.dataset.ic_types),
        target_norm_range=list(ns.target_norm_range),
        zero_mean=bool(ns.zero_mean),
        rng=rng,
        ic_kwargs={
            "max_mode": int(cfg.dataset.max_fourier_mode),
            "amplitude_scale": float(cfg.dataset.fourier_amplitude_scale),
            "n_bumps": int(cfg.dataset.n_gaussian_bumps),
            "spectral_decay": float(cfg.dataset.spectral_decay),
            "zero_mean": bool(ns.zero_mean),
        },
        verbose=True,
    )
    dataset["dataset_version"] = f"nu_sweep_v1_test_extra_{nu_slug(nu)}"
    dataset["protocol"] = (
        f"nu_sweep_v1_test_extra_{nu_slug(nu)}: nu={nu:g} unique, 1 seule valeur; "
        f"target_l2_norm log-uniform{list(ns.target_norm_range)}; zero_mean=True toutes "
        f"familles; Nx={int(ns.Nx)}, dt={float(ns.dt)}, save_every={int(ns.save_every)}, "
        f"{str(ns.time_scheme)}, L={float(ns.L)}, T={float(ns.T)}, 101 snapshots; "
        "seeds disjoints du bloc 100000+ (prereg manuscript/prereg_nu002_strate_test.md)"
    )
    return dataset


def audit(dataset: dict, cfg, nu: float) -> dict:
    ns = cfg.nu_sweep
    gates = b.evaluate_nu_sweep_gates(
        dataset,
        gate_monotone_frac=float(ns.gate_monotone_frac),
        gate_monotone_atol_rel=float(ns.gate_monotone_atol_rel),
        gate_pass_fraction=float(ns.gate_pass_fraction),
        gate_max_abs_u=float(ns.gate_max_abs_u),
        gate_dedt_rel_err=float(ns.gate_dedt_rel_err),
        gate_max_cell_peclet=float(ns.gate_max_cell_peclet),
    )
    summary = b.summarize_dataset_nu_sweep(dataset)
    return {"nu": float(nu), "gates": gates, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nu", type=float, default=0.02, help="viscosité de la strate extra (défaut 0.02, prereg)")
    ap.add_argument("--n", type=int, default=150, help="nb de trajectoires à générer (0 = mode pilote seul)")
    ap.add_argument("--pilot", type=int, default=20, help="nb de trajectoires du pilote (0 = pas de pilote)")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "test_extra_data_nu002.npz",
                    help="npz de sortie (refus si existe)")
    ap.add_argument("--report", type=Path, default=REPO / "outputs" / "logs" / "data_generation_report_nu_extra_nu0p02.json",
                    help="rapport JSON (refus si existe)")
    ap.add_argument("--seed-base", type=int, default=100000, help="base du bloc de seeds (défaut 100000)")
    ap.add_argument("--smoke", action="store_true", help="smoke : 2 trajectoires, écriture UNIQUEMENT dans --smoke-dir")
    ap.add_argument("--smoke-dir", type=Path, default=Path("/tmp/nu_extra_smoke"),
                    help="répertoire du smoke (JAMAIS outputs/)")
    args = ap.parse_args()

    cfg = get_default_config()
    nu = float(args.nu)

    if args.smoke:
        args.smoke_dir.mkdir(parents=True, exist_ok=True)
        out = args.smoke_dir / f"test_extra_smoke_{nu_slug(nu)}.npz"
        report = args.smoke_dir / f"smoke_report_{nu_slug(nu)}.json"
        if out.exists() or report.exists():
            print(f"[SMOKE] fichiers déjà présents, abandon (aucune écrasement) : {out}")
            return 1
        rng = np.random.default_rng(args.seed_base + 999999)
        dataset = build_dataset(cfg, nu, 2, rng)
        aud = audit(dataset, cfg, nu)
        b.save_dataset_npz(out, dataset)
        rep = {"smoke": True, "out": str(out), "sha256": sha256_file(out), **aud}
        report.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        print(f"[SMOKE] npz -> {out} (hors outputs/) ; rapport -> {report}")
        print(f"[SMOKE] gates_passed = {aud['gates']['gates_passed']} ; "
              f"max|u| = {aud['gates']['max_abs_u']:.4f} ; max cell Peclet = {aud['gates']['diagnostics']['max_cell_peclet']:.4f}")
        # preuve qu'aucun fichier de outputs/ n'a été écrit/modifié par le smoke
        print(f"[SMOKE] outputs/ non touché : écriture limitée à {args.smoke_dir}")
        return 0 if aud["gates"]["gates_passed"] else 2

    # --- mode réel (feu vert requis, côté Windows) ---
    if args.out.exists():
        print(f"[ABORT] {args.out} existe déjà : aucune écrasement (règle permanente).")
        return 1
    if args.report.exists():
        print(f"[ABORT] {args.report} existe déjà : aucune écrasement.")
        return 1

    existing = existing_trajectory_seeds()
    rng = np.random.default_rng(args.seed_base)

    if args.pilot > 0:
        pilot_seeds = rng.integers(0, 2 ** 31 - 1, size=args.pilot)
        if np.intersect1d(pilot_seeds, existing).size:
            print("[ABORT] overlap seeds pilote avec les 900 existants.")
            return 1
        pilot_rng = np.random.default_rng(args.seed_base + 777)
        pilot = build_dataset(cfg, nu, args.pilot, pilot_rng)
        aud = audit(pilot, cfg, nu)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"pilot": True, "pilot_n": args.pilot, **aud}, indent=2, default=str), encoding="utf-8")
        print(f"[PILOTE] {args.pilot} traj, gates_passed = {aud['gates']['gates_passed']} -> {args.report}")
        if not aud["gates"]["gates_passed"]:
            print("[PILOTE] NO-GO : ne pas lancer la génération complète.")
            return 2

    n = args.n
    if n <= 0:
        print("[OK] mode pilote seul (--n 0).")
        return 0
    seeds = rng.integers(0, 2 ** 31 - 1, size=n)
    if np.intersect1d(seeds, existing).size:
        print("[ABORT] overlap seeds avec les 900 existants.")
        return 1
    print(f"[GEN] {n} trajectoires à nu={nu:g}, seeds base {args.seed_base}, disjoint OK ({existing.size} existants).")
    dataset = build_dataset(cfg, nu, n, rng)
    aud = audit(dataset, cfg, nu)
    b.save_dataset_npz(args.out, dataset)
    rep = {
        "smoke": False, "nu": nu, "n": n, "seed_base": args.seed_base,
        "n_existing_seeds": int(existing.size), "overlap": False,
        "out": str(args.out), "sha256": sha256_file(args.out),
        "dataset_version": dataset["dataset_version"], "protocol": dataset["protocol"],
        **aud,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"[GEN] npz -> {args.out} ; sha256 = {rep['sha256']}")
    print(f"[GEN] gates_passed = {aud['gates']['gates_passed']} ; rapport -> {args.report}")
    print("[GEN] GO" if aud["gates"]["gates_passed"] else "[GEN] NO-GO (gates)")
    return 0 if aud["gates"]["gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
