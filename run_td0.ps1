$env:PYTHONUNBUFFERED = "1"
foreach ($proto in 'heldout_nu','heldout_nu005') {
  foreach ($mode in 'energy_centered','none') {
    foreach ($seed in 42,7,123,2024,999) {
      $log = "outputs/logs/td_run_td0_${proto}_z${mode}_s${seed}.log"
      python -u main_train_td.py --encoder td0 --protocol $proto --z-target-mode $mode --seed $seed 2>&1 |
        Tee-Object -Append -FilePath $log
      if ($LASTEXITCODE -ne 0) {
        "$(Get-Date -Format s) ECHEC td0 $proto $mode s$seed (code $LASTEXITCODE)" |
          Add-Content outputs/logs/td_failures.log
      }
    }
  }
}

