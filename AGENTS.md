# EcoSort contributor guide

- Keep production Python code in `src/ecosort`; keep the Streamlit UI thin in `app/app.py`.
- Never add `data/`, checkpoints, or generated plots to Git. Use paths from `config/default.yaml` or CLI options.
- Run `pytest` after changing data, model, inference, or UI utility code. Keep CPU inference functional.
- Prefer type hints, docstrings on public functions, deterministic behavior where practical, and `pathlib.Path` over hardcoded paths.
- The supported dataset layout is `data/raw/<class_name>/<image files>`; run `python -m ecosort.data validate` before training.
