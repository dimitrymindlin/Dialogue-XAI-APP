# Dialogue-XAI App

Dialogue-XAI is a backend-driven application for helping users understand model predictions through multiple explanation styles:
- `static`: precomputed, report-style explanations
- `interactive`: guided question-click exploration
- `chat`: natural-language dialogue, optionally with LLM agents (including MAPE-K variants)

The stack centers on a Flask API (`flask_app.py`), an orchestration layer (`explain/logic.py`), and pluggable XAI methods (SHAP/LIME-based feature influence, DiCE, Anchors, Ceteris Paribus, feature statistics, PDP).

## What This Repo Is For

Use this repository to:
- run user-study style explanation workflows for tabular ML models
- compare explanation interaction modes (`static`, `interactive`, `chat`)
- test adaptive explanation agents (MAPE-K and conversational variants)
- generate and analyze experiment data

## Architecture At A Glance

- API entrypoint: `flask_app.py`
- Core orchestration: `explain/logic.py` (`ExplainBot`)
- XAI dispatch: `explain/action.py` (`question_id -> explanation method`)
- LLM/agent implementations: `llm_agents/`
- Configs: `global_config.gin`, `configs/*.gin`
- Frontend templates/static assets: `templates/`
- Cached explanation artifacts/logs: `cache/`

Detailed flow diagrams and method-dispatch maps:
- [Backend architecture](docs/backend-architecture.md)
- [Environment configuration and agent variants](docs/environment-configuration.md)

## Modes And User Flow

1. Frontend calls `GET /init` with `user_id`, `study_group`, and `ml_knowledge`.
2. Backend creates one `ExplainBot` per `user_id`.
3. `GET /get_train_datapoint` returns the next instance.
4. User sends prediction via `POST /set_user_prediction`.
5. User explores explanations by:
- `POST /get_response_clicked` (interactive button/click flow), or
- `POST /get_response_nl` (chat/NL flow; streaming supported).
6. Session cleanup via `DELETE /finish`.

## Quickstart (Local)

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Required environment variables

At minimum set:
- `ML_EXECUTOR_THREADS=2` (or another integer)
- `OPENAI_API_KEY=...` (required if using OpenAI-backed intent/agent paths)

### 3) Choose config and agent mode

- Default dataset/runtime config is loaded from `global_config.gin`.
- Override dataset config file with:
```bash
export XAI_CONFIG_PATH=./configs/adult-config.gin
```
- Select agent family with:
```bash
export XAI_USE_LLM_AGENT=False
```
or e.g.
```bash
export XAI_USE_LLM_AGENT=mape_k_openai_unified
```

### 4) Run backend

Development:
```bash
python flask_app.py
```

Production-style local run:
```bash
gunicorn --workers 4 --worker-class gevent --worker-connections 20 --timeout 140 --bind 0.0.0.0:4000 flask_app:app
```

## Configuration Model

- `global_config.gin`
  - base config selector (`GlobalArgs.config`)
  - API base URL
- `configs/*-config.gin`
  - model/data file paths
  - class names and dataset descriptions
  - explanation/cache tuning
  - behavior toggles (intent recognition, dialogue manager, followups, submodular pick)
- env vars (runtime override layer)
  - `XAI_CONFIG_PATH`: choose gin file at runtime
  - `XAI_USE_LLM_AGENT`: choose agent type
  - `ML_EXECUTOR_THREADS`: thread pool size for expensive calls

## Core Endpoints

- `GET /init`
- `GET /get_train_datapoint`
- `GET /get_test_datapoint`
- `GET /get_final_test_datapoint`
- `POST /set_user_prediction`
- `POST /get_response_clicked`
- `POST /get_response_nl`
- `GET /get_user_model`
- `POST /set_user_model`
- `DELETE /finish`

## Testing

Run all tests:
```bash
pytest
```

Focused MAPE-K-2 test:
```bash
python -m pytest llm_agents/mape_k_2_components/test_unified_agent.py
```

Integration/system checks:
```bash
python mock_backend_testing.py
python explainbot_direct_test.py
```

## Caching And Logs

- Explanation caches are stored under `cache/*.pkl`.
- Per-instance XAI report cache is managed by `explain/xai_cache_manager.py`.
- LLM execution traces/logging are stored in:
  - `cache/llm_executions.csv`

Monitor in real time:
```bash
tail -f cache/llm_executions.csv
```

## Repo Layout

- `flask_app.py`: Flask routing/session orchestration
- `explain/`: ExplainBot, explanation actions, dialogue manager, cache manager
- `llm_agents/`: conversational + MAPE-K agent implementations
- `create_experiment_data/`: datapoint generation and experiment utilities
- `configs/`: dataset/runtime gin configs
- `data/`: model artifacts, datasets, mappings, response templates
- `templates/`: frontend template assets
- `experiment_analysis/`: study analysis pipeline and outputs
- `docs/`: architecture/configuration docs

## Notes For New Teammates

- Start from `flask_app.py` (request lifecycle), then `explain/logic.py` (bot orchestration), then `explain/action.py` (explanation dispatch).
- If behavior seems wrong, first verify:
1. loaded config file (`XAI_CONFIG_PATH`)
2. agent mode (`XAI_USE_LLM_AGENT`)
3. study group passed to `/init` (`static` vs `interactive` vs `chat`)
4. cache freshness in `cache/`
