# Sapia

A ~1.2B-parameter language model, pretrained from scratch on a curated English mix (FineWeb-Edu plus conversational data) on a single GH200.

Status: model and data preparation written; training loop next. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the proposed architecture and [docs/DATA.md](docs/DATA.md) for the data mix.

Sapia is an internal model: one user, weights not released.

Inspired by `modal-lm` (the 257M `gpt250m` model).

## Data preparation

```bash
pip install -e ".[dev]"
python -m sapia.prepare_mix --preset tiny            # ~2M tokens, one file per source (minutes)
python -m sapia.prepare_mix --total_tokens 15e9      # the real mix (resumable; rerun to continue)
pytest
```

Prepared data goes to `data/` (git-ignored). Keep it outside OneDrive for real runs.
