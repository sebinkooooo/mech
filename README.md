# Mech - Temporal Gait Transformer

This repository contains our gait classification pipeline for smart insole data.

## What to run

- `train_tgt_v2.ipynb` - main training notebook (recommended)
- `sebi_v2.py` - model and preprocessing utilities used by `train_tgt_v2.ipynb`
- `cross_val_tgt.ipynb` - file-level cross-validation notebook

## Important note (v2)

`v2` is the current version and uses:

- Euler-angle feature set (`ele_22`, `ele_23`, `ele_24`) to match the shared evaluation template
- Mixup enabled (`MIXUP_ALPHA = 0.2`)

This is the version to use when exporting the `.pth` model for evaluation.

## Main files

- `sebi.py` - earlier utility module (quaternion-feature variant)
- `sebi_v2.py` - current utility module (Euler-feature variant, mixup-compatible pipeline)
- `train_tgt.ipynb` - earlier training notebook
- `train_tgt_v2.ipynb` - current training notebook
- `cross_val_tgt.ipynb` - k-fold file-level validation
- `paper/` - paper draft and LaTeX assets
- `report/` - compact report and appendix with source listing

## Reproducibility links

- Repo: [github.com/sebinkooooo/mech](https://github.com/sebinkooooo/mech.git)
- Training run notebook: [Google Drive run](https://drive.google.com/file/d/1iLZLUPbs637IBlTrdomt-Pahzhj9ZaGV/view?usp=sharing)
