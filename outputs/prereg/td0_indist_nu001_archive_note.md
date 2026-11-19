# Note d'archivage — runs `td0_indist_nu001_*` (anciennement `td0_heldout_nu_*`)

Les 10 runs `td0_indist_nu001_{zenergy_centered_,}s{42,7,123,2024,999}` ont été
**entraînés sur ν=0.01 seul par erreur de filtre** (`main_train_td.py:48` passait
`heldout_nu_values=(0.01,)` au lieu de `(0.05, 0.2)` exigé par le prereg
`td_encoder_prereg.md:40` et `config.py:417`). Ils constituent un **contrôle
in-distribution** à ν=0.01, **hors verdict** : aucune métrique de ces dossiers ne
doit être lue comme un résultat held-out.

**Preuve** : leur `z_target_standardizer` (mean=0.16841002, std=0.23469113) est
reproduit à 8 décimales en refaisant le fit sur le train filtré à ν=0.01 seul
(recette `ml_datasets_experiment_c.py:578-583,605-611`), et le champ
`heldout_nu_values=[0.01]` est enregistré dans leurs metrics. Les runs
`window_w25_heldout_nu_*` (Step 6) ont, eux, `heldout_nu_values=[0.05, 0.2]`
(conformes).

**Correction** : commit `36ba701` (module `td_protocols.py`, source de vérité
unique ; garde-fou `expected_train_nu` qui refuse tout run dont l'ensemble ν
effectif diffère du prereg ; test `python3 td_protocols.py --check`).

**Relance** : les 10 runs `td0_heldout_nu_*` corrigés sont relancés sous leur nom
d'origine ; l'évaluateur (`main_evaluate_td.py:107-110`) ne lit que les noms
`td{bras}_{protocole}[_z{mode}]_s{seed}`, donc les dossiers archivés sont ignorés
et ne bloquent pas la logique de saut de `main_train_td.py`.

Rien n'a été supprimé : `git mv` uniquement (contenus bit-identiques).
