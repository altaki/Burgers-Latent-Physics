"""
main_train_latent_seq.py
========================

Point d'entrée de window-A (Architecture A-Sequence, Step 5) : Architecture A
(`LatentDynamicsAutoencoder`) avec encodeur Conv1D sur fenêtre de `window_size`
snapshots, entraînée sur nu_sweep_v1, évaluée contre nu*D à nu held-out.

Usage (depuis la racine du projet, environnement torch) :

    python main_train_latent_seq.py                          # w=25, standard + heldout_nu, tag "aseq"
    python main_train_latent_seq.py --window-size 50         # autre fenêtre
    python main_train_latent_seq.py --protocols standard     # un seul protocole
    python main_train_latent_seq.py --tag aseq_w50           # rapports suffixés

Experiment A-Physics (Step 6, z ancré sur l'identité d'énergie dE/dt = -νD) :

    python main_train_latent_seq.py --tag aphysics --z-target energy_centered --z-weight 1.0 --mono-weight 0.1
    python main_train_latent_seq.py --tag asup     --z-target supervised      --z-weight 1.0 --mono-weight 0.1
    # runs dans outputs/models/latent_dynamics_autoencoder_seq/window_w25_{protocole}_z{energy_centered,supervised}/

Sorties :
- modèles/métriques : `outputs/models/latent_dynamics_autoencoder_seq/window_w{w}_{protocole}/`
  (best_model.pt, metrics JSON + CSV long, standardizers.json) ;
- résumé + verdict : `outputs/logs/experiment_a_seq_summary[_tag].json`
  (r_holdout, R² held-out, pente de calibration, beats_nu_oracle, critères
  structurels, verdict IDENTIFIABLE / PARTIAL / NOT IDENTIFIABLE) + snapshot config.

Le verdict n'est calculé que si les deux protocoles ont tourné.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import config as config_module


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Window-A (Architecture A-Sequence) : LatentDynamicsAutoencoder + encodeur Conv1D "
                    "sur fenêtre de w snapshots, sur nu_sweep_v1, évaluation z vs nu*D à nu held-out."
    )
    parser.add_argument(
        "--protocols", nargs="*", default=None,
        choices=["standard", "heldout_nu"],
        help="Protocoles à exécuter (défaut: ceux de config.experiment_a_seq).",
    )
    parser.add_argument(
        "--window-size", type=int, default=None, metavar="W",
        help="Taille de la fenêtre temporelle (défaut: config.experiment_a_seq.window_size = 25).",
    )
    parser.add_argument(
        "--tag", type=str, default="aseq",
        help="Suffixe des rapports globaux (défaut 'aseq' -> experiment_a_seq_summary_aseq.json).",
    )
    parser.add_argument(
        "--z-target", type=str, default=None,
        choices=["none", "energy_centered", "supervised"],
        help="Cible physique de z (Experiment A-Physics, Step 6) : 'energy_centered' = nuD estimé "
             "par différence centrée de E (auto-supervisé), 'supervised' = nuD stocké. "
             "Défaut: config.experiment_a_seq.z_target_mode ('none').",
    )
    parser.add_argument(
        "--z-weight", type=float, default=None, metavar="LZ",
        help="Poids λ_z du terme MSE(z_t, cible) (défaut: config.experiment_a_seq.z_weight).",
    )
    parser.add_argument(
        "--mono-weight", type=float, default=None, metavar="LM",
        help="Poids λ_mono du hinge de monotonie (défaut: config.experiment_a_seq.monotonicity_weight). "
             "Recommandé 0.1 pour A-Physics.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Graine (torch/numpy/random + DataLoaders). Apparaît dans le nom du run et des summaries. "
             "Défaut: config.reproducibility.seed.",
    )
    parser.add_argument(
        "--monotone-rtol", type=float, default=None, metavar="RTOL",
        help="Tolérance relative de la fraction monotone (défaut: config.experiment_a_seq.monotone_rtol = 0).",
    )
    parser.add_argument(
        "--no-standardize-z-target", action="store_true",
        help="Désactive la standardisation de la cible z (loss en unités physiques brutes).",
    )
    return parser.parse_args(argv)


def run_a_seq(
    config: Any,
    protocols: Optional[Sequence[str]] = None,
    window_size: Optional[int] = None,
    tag: str = "aseq",
    z_target_mode: Optional[str] = None,
    z_weight: Optional[float] = None,
    mono_weight: Optional[float] = None,
    seed: Optional[int] = None,
    monotone_rtol: Optional[float] = None,
    standardize_z_target: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Exécute window-A pour chaque protocole demandé et écrit le résumé + verdict."""
    import ml_train_experiment_a_seq as mt  # import tardif : requiert torch

    ec = config.experiment_c
    aseq = config.experiment_a_seq

    protocols = list(protocols) if protocols is not None else list(aseq.protocols)
    w = int(window_size) if window_size is not None else int(aseq.window_size)
    if w < 1:
        raise ValueError("`window_size` doit être >= 1.")
    z_mode = z_target_mode if z_target_mode is not None else aseq.z_target_mode
    z_w = float(z_weight) if z_weight is not None else float(aseq.z_weight)
    mono_w = float(mono_weight) if mono_weight is not None else float(aseq.monotonicity_weight)
    seed_val = int(seed) if seed is not None else int(config.reproducibility.seed)
    rtol = float(monotone_rtol) if monotone_rtol is not None else float(aseq.monotone_rtol)
    std_z = bool(standardize_z_target) if standardize_z_target is not None else bool(aseq.standardize_z_target)

    data_dir = Path(config.paths.data_dir)
    models_root = Path(config.paths.models_dir) / aseq.models_subdir
    logs_dir = ensure_dir(config.paths.logs_dir)

    if verbose:
        print("\n" + "=" * 70)
        print("Window-A (Architecture A-Sequence) sur nu_sweep_v1")
        print("=" * 70)
        print(f"- protocoles : {protocols}")
        print(f"- fenêtre    : w={w}")
        print(f"- tag        : {tag}  seed: {seed_val}")
        print(f"- z-target   : {z_mode} (λ_z={z_w}, λ_mono={mono_w}, standardize={std_z}, rtol={rtol})")
        print(f"- sorties    : {models_root}")

    runs: Dict[str, Dict[str, Any]] = {}
    for protocol in protocols:
        run_name = f"window_w{w}_{protocol}"
        if z_mode != "none":
            run_name += f"_z{z_mode}"
        run_name += f"_s{seed_val}"
        output_dir = ensure_dir(models_root / run_name)
        if verbose:
            print(f"\n--- protocole {protocol} -> {output_dir} ---")
        results = mt.train_and_evaluate_a_seq(
            protocol=protocol,
            train_path=data_dir / ec.train_filename,
            val_path=data_dir / ec.val_filename,
            test_path=data_dir / ec.test_filename,
            output_dir=output_dir,
            window_size=w,
            heldout_nu_values=ec.heldout_nu_values,
            batch_size=aseq.batch_size,
            learning_rate=aseq.learning_rate,
            num_epochs=aseq.num_epochs,
            patience=aseq.patience,
            weight_decay=aseq.weight_decay,
            latent_dim=aseq.latent_dim,
            conv_channels=aseq.conv_channels,
            kernel_size=aseq.kernel_size,
            conv_stride=aseq.conv_stride,
            trunk_dims=aseq.trunk_dims,
            decoder_hidden_dims=aseq.decoder_hidden_dims,
            predictor_hidden_dims=aseq.predictor_hidden_dims,
            scalar_head_hidden_dims=aseq.scalar_head_hidden_dims,
            activation=aseq.activation,
            dropout=aseq.dropout,
            reconstruction_weight=aseq.reconstruction_weight,
            prediction_weight=aseq.prediction_weight,
            monotonicity_weight=mono_w,
            variance_weight=aseq.variance_weight,
            min_latent_variance=aseq.min_latent_variance,
            monotonicity_mode=aseq.monotonicity_mode,
            prediction_mode=aseq.prediction_mode,
            z_target_mode=z_mode,
            z_weight=z_w,
            standardize_z_target=std_z,
            monotone_rtol=rtol,
            eval_times=aseq.eval_times,
            seed=seed_val,
            device=config.ml.device,
            num_workers=0,
            print_every=10,
            verbose=verbose,
        )
        runs[protocol] = {
            "output_dir": str(output_dir),
            "window_size": w,
            "checkpoint_path": results["checkpoint_path"],
            "best_epoch": results["best_epoch"],
            "best_val_loss": results["best_val_loss"],
            "stopped_early": results["stopped_early"],
            "results": results,
        }

    thresholds = {
        "r_holdout_min": aseq.r_holdout_min,
        "r2_holdout_min": aseq.r2_holdout_min,
        "slope_min": aseq.slope_min,
        "slope_max": aseq.slope_max,
        "r_holdout_fail": aseq.r_holdout_fail,
        "rank_min": aseq.rank_min,
        "rec_rmse_max": aseq.rec_rmse_max,
        "monotone_min": aseq.monotone_min,
    }

    verdict_version = "v2_aphysis" if z_mode != "none" else "v1"
    verdict: Dict[str, Any] = {}
    if "standard" in runs and "heldout_nu" in runs:
        verdict["window_a"] = mt.verdict_a_seq(
            runs["standard"]["results"], runs["heldout_nu"]["results"], thresholds,
            z_target_mode=z_mode, verdict_version=verdict_version,
        )

    summary: Dict[str, Any] = {
        "experiment": "experiment_a_seq",
        "tag": tag,
        "dataset_version": "nu_sweep_v1",
        "seed": seed_val,
        "window_size": w,
        "z_target_mode": z_mode,
        "z_weight": z_w,
        "monotonicity_weight": mono_w,
        "standardize_z_target": std_z,
        "monotone_rtol": rtol,
        "verdict_version": verdict_version,
        "protocols": protocols,
        "runs": {
            p: {k: v for k, v in info.items() if k != "results"}
            for p, info in runs.items()
        },
        "verdict": verdict,
        "thresholds": thresholds,
    }

    def _tagged(filename: str) -> str:
        # un summary par (tag, graine) : évite l'écrasement par la dernière invocation
        suffix = f"_{tag}_s{seed_val}" if tag else f"_s{seed_val}"
        return filename.replace(".json", f"{suffix}.json")

    summary_path = logs_dir / _tagged(aseq.summary_filename)
    save_json(summary_path, summary)
    save_json(logs_dir / _tagged(aseq.config_snapshot_filename), config.to_dict())

    if verbose:
        print("\n" + "=" * 70)
        print("Résumé window-A")
        print("=" * 70)
        for p, info in runs.items():
            print(f"- {p:10s} best_epoch={info['best_epoch']} best_val_loss={info['best_val_loss']:.6e} "
                  f"stopped_early={info['stopped_early']}")
        if "window_a" in verdict:
            d = verdict["window_a"]
            print(f"- verdict z vs nu*D (nu=0.01 held-out) = {d['verdict']}")
            print(f"    r_holdout={d['r_holdout']:.3f} R2_holdout={d['r2_holdout_nu001']:.3f} "
                  f"cal_slope={d['calibration_slope_nu001']:.3f} beats_nu_oracle={d['beats_nu_oracle']}")
            print(f"    rang={d['effective_rank']:.0f} rec_rmse={d['reconstruction_rmse_test']:.4f} "
                  f"monotone={d['monotone_fraction_test']:.4f} structural_ok={d['structural_ok']}")
        print(f"Résumé sauvegardé : {summary_path}")

    return summary


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    config = config_module.get_default_config(create_dirs=True)
    return run_a_seq(
        config,
        protocols=args.protocols,
        window_size=args.window_size,
        tag=args.tag,
        z_target_mode=args.z_target,
        z_weight=args.z_weight,
        mono_weight=args.mono_weight,
        seed=args.seed,
        monotone_rtol=args.monotone_rtol,
        standardize_z_target=False if args.no_standardize_z_target else None,
        verbose=True,
    )


if __name__ == "__main__":
    main()
