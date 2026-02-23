# Backend Architecture: Information Flow and XAI Execution

This document explains how data flows through the backend and how configuration controls behavior.

## 1) End-to-end flow

```mermaid
flowchart TD
    A["Frontend UI<br/>static / interactive / chat"] --> B["Flask API<br/>flask_app.py"]

    B -->|GET /init| C["Create ExplainBot instance<br/>per user_id in bot_dict"]
    C --> D["Load gin config<br/>global_config.gin + XAI_CONFIG_PATH override"]
    C --> E["Load model + datasets<br/>Conversation"]
    C --> F["Load TemplateManager<br/>feature names / tooltips / units"]
    C --> G["Preload XAI engines + caches<br/>MegaExplainer, DiCE, Anchor, CP, PDP, FeatureStats"]
    C --> H["Optional LLM agent<br/>selected by XAI_USE_LLM_AGENT"]
    C --> I["DialogueManager<br/>(optional active policy)"]

    B -->|GET /get_*_datapoint| J["ExplainBot.get_next_instance"]
    J --> K{study_group}
    K -->|static| L["compute_explanation_report<br/>full static report returned with datapoint"]
    K -->|interactive/chat| M["return datapoint only"]

    B -->|POST /set_user_prediction| N["Store user guess<br/>return initial prompt"]

    B -->|POST /get_response_clicked| O["update_state_new(question_id, feature_id)"]
    B -->|POST /get_response_nl| P["update_state_from_nl(...)"]

    O --> Q["DialogueManager.update_state<br/>resolve method + feature"]
    P --> R{use_llm_agent?}
    R -->|yes| S["agent.answer_user_question(_stream)"]
    R -->|no| Q

    Q --> T["run_action_new<br/>question_id -> explanation function"]
    T --> U["XAI method from Conversation<br/>(uses preloaded explainers/cache)"]
    U --> V["HTML/text response + optional followups"]
    S --> V
    V --> B
    B --> A
```

## 2) Explanation dispatch (how methods are called)

`ExplainBot.update_state_new(...)` calls `run_action_new(...)` in `explain/action.py`, which dispatches by `question_id`:

```mermaid
flowchart LR
    QID[question_id] --> D1[top3Features / least3Features]
    QID --> D2[shapAllFeatures / shapAllFeaturesPlot]
    QID --> D3[counterfactualAnyChange / counterfactualSpecificFeatureChange]
    QID --> D4[anchor]
    QID --> D5[ceterisParibus]
    QID --> D6[featureStatistics]
    QID --> D7[globalPdp]
    QID --> D8[modelConfidence]
    QID --> D9[followup* / whyExplanation / greeting]

    D1 --> M1[explain_local_feature_importances]
    D2 --> M2[explain_feature_importances_as_plot]
    D3 --> M3[explain_cfe / explain_cfe_by_given_features]
    D4 --> M4[explain_anchor_changeable_attributes_without_effect]
    D5 --> M5[explain_ceteris_paribus]
    D6 --> M6[explain_feature_statistic]
    D7 --> M7[explain_pdp]
    D8 --> M8[explain_model_confidence]
```

All of these functions read explainer objects from `Conversation` that were initialized in `ExplainBot.load_explanations(...)`.

## 3) What settings influence

### Runtime/env settings
- `XAI_CONFIG_PATH`: chooses dataset-specific gin config file; overrides `GlobalArgs.config`.
- `XAI_USE_LLM_AGENT`: selects agent family (`mape_k`, `mape_k_2`, `conversational`, etc.) or disables LLM agent.
- `ML_EXECUTOR_THREADS`: size of backend thread pool for heavy inference/explanation calls.

### Gin settings (dataset and behavior)
From `configs/*-config.gin` (example `configs/adult-config.gin`):
- Data/model wiring: model path, dataset path, categorical mapping, encoded mapping.
- Semantics shown in UI/prompts: class names, instance naming, dataset descriptions.
- Behavior toggles:
  - `ExplainBot.use_intent_recognition`
  - `ExplainBot.use_active_dialogue_manager`
  - `ExplainBot.use_static_followup`
  - `ExplainBot.submodular_pick`
- XAI/cache parameters:
  - `MegaExplainer.cache_location`
  - `TabularDice.cache_location`
  - `TabularAnchor.cache_location`
  - `CeterisParibus.cache_location`
  - `PdpExplanation.cache_location`
  - `DiverseInstances.*`, `TestInstances.*`, `Explanation.max_cache_size`

### Study mode switch (static vs interactive/chat)
`study_group` is passed on `/init`:
- `static`: on `/get_train_datapoint`, backend attaches `static_report` (precomputed multi-method report).
- `interactive` and `chat`: user asks incrementally; backend serves per-question explanations.
- `chat` with LLM agent: NL goes through agent path (MAPE-K/conversational variants).

## 4) Role of templates

`TemplateManager` (`data/response_templates/template_manager.py`) standardizes:
- feature display names
- tooltips/units
- decoding categorical/encoded values

This layer affects how explanation outputs are rendered, but not model predictions themselves.

## 5) Where cache is used

- Method-level caches: each explainer persists artifacts to `./cache/*.pkl`.
- Cross-method per-instance cache: `XAICacheManager` stores full XAI reports (`./cache/{dataset}-xai-reports.pkl`) used for fast LLM-agent datapoint initialization.

## 6) Fast mental model

1. `flask_app.py` manages sessions/endpoints and delegates all intelligence to `ExplainBot`.
2. `ExplainBot` builds a `Conversation` with model+data+all explainers once, then serves each turn.
3. `run_action_new(question_id, feature_id)` is the central dispatch for explanation method execution.
4. Settings decide: dataset wiring, mode (static/interactive/chat), and whether NL is rule-based or LLM-agent based.
