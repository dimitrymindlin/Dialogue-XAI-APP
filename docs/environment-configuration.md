# Environment Variable Configuration

## LLM Agent Selection (`XAI_USE_LLM_AGENT`)

`XAI_USE_LLM_AGENT` controls which chat agent backend is instantiated in `ExplainBot` (`explain/logic.py`).

- Variable: `XAI_USE_LLM_AGENT`
- Priority: environment variable overrides gin config
- Disable value: `False` (string, case-insensitive)

If not set, the app falls back to the gin value (`ExplainBot.use_llm_agent`).

## Supported Values And Routing

These values are currently routed in code and map to the following classes:

| Value(s) | Backend class | Stack |
|---|---|---|
| `o1` | `llm_agents.o1_agent.openai_o1_agent.XAITutorAssistant` | OpenAI |
| `mape_k`, `mape_k_4` | `llm_agents.mape_k_component_mixins.MapeK4BaseAgent` | LlamaIndex workflow |
| `mape_k_2` | `llm_agents.mape_k_component_mixins.MapeK2BaseAgent` | LlamaIndex workflow |
| `unified_mape_k`, `mape_k_unified` | `llm_agents.unified_agent.MapeKUnifiedBaseAgent` | LlamaIndex workflow |
| `mape_k_approval_2` | `llm_agents.mape_k_component_mixins.MapeKApprovalBaseAgent` | LlamaIndex workflow |
| `mape_k_approval`, `mape_k_approval_4` | `llm_agents.mape_k_component_mixins.MapeKApproval4BaseAgent` | LlamaIndex workflow |
| `mape_k_openai` | `llm_agents.openai_mapek_agent.MapeK4OpenAIAgent` | OpenAI Agents SDK |
| `mape_k_openai_2` | `llm_agents.openai_mapek_agent.MapeK2OpenAIAgent` | OpenAI Agents SDK |
| `mape_k_openai_unified` | `llm_agents.openai_mapek_agent.MapeKUnifiedOpenAIAgent` | OpenAI Agents SDK |
| `simple_openai`, `simple_openai_agent` | `llm_agents.simple_openai_agent.SimpleOpenAIAgent` | OpenAI Agents SDK |
| `conversational` | `llm_agents.simple_conv_agent.ConversationalStreamAgent` | LlamaIndex workflow |

Any other value raises `ValueError("Unknown agent type: ...")` during bot init.

## Variant Differences (What Changes)

### By MAPE-K decomposition
- `mape_k` / `mape_k_4`: explicit 4-phase Monitor -> Analyze -> Plan -> Execute.
- `mape_k_2`: combined Monitor+Analyze and Plan+Execute.
- `unified_mape_k` / `mape_k_unified`: single prompt handling all phases.
- `mape_k_approval_2`: adaptive 2-step flow; first turn creates plan, later turns approve/adjust existing plan.
- `mape_k_approval_4`: 4-step flow with explicit plan-approval phase.

### Planning-time behavior
- Approval variants (`mape_k_approval_2`, `mape_k_approval`, `mape_k_approval_4`) are designed to reuse an existing explanation plan and only approve/reorder/adjust it on later turns.
- In practice, this typically lowers planning overhead after the first turn compared with rebuilding plans from scratch each turn.

### By runtime/provider family
- `*_openai*` and `simple_openai*`: OpenAI Agents SDK-based implementations.
- Non-`openai` variants above: LlamaIndex workflow + local mixins.

### Non-MAPE-K options
- `conversational`: dialogue agent without explicit MAPE-K planning phases. In the paper, this variant is referred to as **Baseline LLM**.
- `simple_openai`: direct QA agent with explanation context, no MAPE-K phase logic.

## Recommended Defaults

- Most capable OpenAI MAPE-K single-call mode: `mape_k_openai_unified`
- More interpretable step-by-step orchestration: `mape_k_openai` or `mape_k`
- Fast/simple baseline: `simple_openai` or `conversational`
- Disable LLM agent (use existing non-agent interaction paths): `False`

## Usage Examples

In `.env`:

```bash
XAI_USE_LLM_AGENT=mape_k_openai_unified
```

Disable LLM agent:

```bash
XAI_USE_LLM_AGENT=False
```

Simple direct OpenAI agent:

```bash
XAI_USE_LLM_AGENT=simple_openai
```

## Related Runtime Settings

- `XAI_CONFIG_PATH`: overrides which `configs/*.gin` file is loaded.
- `ML_EXECUTOR_THREADS`: thread pool size for heavy compute calls in `flask_app.py`.
