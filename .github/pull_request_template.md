<!--
Control video report — MLflow + Hydra topic. Fill every field.
The PR description is part of your submission.
-->

**Video link:** <!-- up to 5 minutes -->
**Video duration:** <!-- mm:ss -->
**Level claimed:** <!-- core (4 checks) / +DVC bonus -->

## Experiment results (must match method-note.md §6 exactly)

| Run | Command | accuracy | f1 |
|-----|---------|----------|-----|
| E1 | `python train.py` | 0.9561 | 0.9655 |
| E2 | `python train.py model=lr` | | |
| E3 | `python train.py model.n_estimators=300` | | |
| E4 | `python train.py seed=7` | | |

## Verification commands

```
python -m pytest -m basic -v
python -m pytest -m advanced -v
python -m pytest -m expert -v
```

## Checklist

- [ ] All video checks pass (identifiability, reproducibility, CLI override, comparison)
- [ ] Reference metrics match exactly
- [ ] REPORT.md is filled in
- [ ] No video files, mlruns/, or model.pkl committed
