# Archive des sorties de l'évaluateur B1 et chronologie

- `b1_interpolation_results_pre166c1ff.json` : sortie produite avec le code de `2fa9b98`
  (avant la correction de `calibration_slope` en `166c1ff`) : C3_slope = 0.8262 (pente
  inversée z~y), C5 = 3.3723 (participation ratio). Exécution de production entre
  2026-11-12 11:11:55 (commit 2fa9b98) et 11:15:08 (commit 166c1ff).
- `b1_interpolation_results_pre5e6f24e.json` : sortie produite avec le code de `166c1ff`
  (avant la correction de C5 en rang 95 % en `5e6f24e`) : C3_slope = 1.1531 (bonne
  définition), C5 = 3.3723 (participation ratio). Exécution entre 11:15:08 et
  11:25:07 (mtime du fichier de production 2026-11-12 11:25:07).
- Production finale (code `5e6f24e`, C5 = rang 95 % = 5.667) : `outputs/logs/b1_interpolation_results.json`, mtime 2026-11-12 11:25:07, commit artefacts `a6cb108`.
- Corrections effectuées après une première lecture des résultats, dans le sens du
  texte commité fc1075b, contre l'intérêt apparent du résultat (C5 : PR s42 = 2.757 < 3
  aurait fait échouer la condition ; rang 95 % = 5.667 la fait passer). Verdict inchangé
  (PARTIAL, C6 seul échec) dans les trois versions.
- pip freeze de l'environnement Windows : NON CONSIGNÉ — inaccessible depuis le sandbox
  de cette session. À produire côté env VS Code :
  `pip freeze > outputs/logs/b1_eval_archive/pip_freeze_windows.txt`
