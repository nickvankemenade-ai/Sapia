# Sapia

A 1.05B-parameter language model (v1: 28 layers x 1792, Muon optimizer), pretrained from scratch on a curated English mix (FineWeb-Edu plus conversational data) on a single GPU. Design: [docs/DESIGN_V1.md](docs/DESIGN_V1.md).

Status: complete pipeline (data prep, training, export). See [docs/RUNBOOK.md](docs/RUNBOOK.md) to run it on a GPU server. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the proposed architecture and [docs/DATA.md](docs/DATA.md) for the data mix.

Sapia is an internal model: one user, weights not released.

Inspired by `modal-lm` (the 257M `gpt250m` model).

## Data preparation

```bash
pip install -e ".[dev]"
python -m sapia.prepare_mix --preset tiny            # ~2M tokens, one file per source (minutes)
python -m sapia.prepare_mix --total_tokens 7.5e9     # the real mix (resumable; rerun to continue)
pytest
```

Prepared data goes to `data/` (git-ignored). Keep it outside OneDrive for real runs.

## Training on Kaggle TPUs

`python -m sapia.train --tpu ...`: see [docs/KAGGLE.md](docs/KAGGLE.md) (sessions of ~9 hours that continue from each other).
