# Structured MAPE-K Implementation

This implementation enhances the unified MAPE-K agent by adding structured output using Pydantic models. The structured approach ensures that the LLM's responses adhere to a specific schema, making the system more reliable and consistent.

## Key Components

### MAPE_K_ResultModel

The central Pydantic model that structures the output of the MAPE-K workflow:

```python
class MAPE_K_ResultModel(BaseModel):
    # Monitor stage
    monitor_reasoning: str 
    explicit_understanding_displays: List[str]
    mode_of_engagement: str
    
    # Analyze stage
    analyze_reasoning: str
    model_changes: List[ModelChange]
    
    # Plan stage
    plan_reasoning: str
    new_explanations: List[NewExplanationModel]
    explanation_plan: List[ChosenExplanationModel]
    next_response: List[ExplanationTarget]
    
    # Execute stage
    execute_reasoning: str
    response: str
```

### Sub-models

- `NewExplanationModel`: For new explanations to be added
- `ChosenExplanationModel`: For explanations to be included in the plan
- `ExplanationTarget`: For specific explanation targets in the response
- `CommunicationGoal`: For communication goals related to explanations
- `ModelChange`: For changes to the user model

## How It Works

1. The UnifiedMapeKAgent processes user questions through a single LLM call
2. The LLM generates a structured JSON response following the MAPE_K_ResultModel schema
3. The agent parses this response into Pydantic models, ensuring type safety
4. The structured data is used to update the user model and generate responses

## Using the Structured MAPE-K Agent

To use the structured MAPE-K agent in the application:

1. Set the `use_llm_agent` parameter to "structured_mape_k" in the ExplainBot initialization
2. Or use the provided Gin configuration file: `configs/structured_mape_k.gin`

Example:
```python
from explain.logic import ExplainBot

bot = ExplainBot(
    # other parameters...
    use_llm_agent="structured_mape_k"
)
```

## Testing

You can test the structured MAPE-K implementation using:

```bash
python test_structured_mape_k.py
```

## Benefits of Structured Output

1. **Type Safety**: Ensures responses conform to expected types
2. **Consistency**: Maintains consistent structure across different LLM responses
3. **Reliability**: Reduces errors from parsing unstructured text
4. **Maintainability**: Makes it easier to extend and modify the agent's behavior

## Future Improvements

- Add more detailed validation rules to the Pydantic models
- Enhance error handling for cases where the LLM doesn't follow the schema
- Expand the models to capture more nuanced aspects of the MAPE-K workflow 