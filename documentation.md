# Dialogue-XAI-APP: Project Structure and Functionality

## Overview

Dialogue-XAI-APP is a system designed to explain machine learning model predictions through interactive dialogue. The system uses various Explainable AI (XAI) methods to help users understand ML model decisions and adapts explanations based on the user's level of understanding.

## Project Structure

```
Dialogue-XAI-APP/
├── llm_agents/                  # Main agent modules
│   ├── api/                     # API integration
│   ├── frontend/                # Frontend components
│   ├── workflow_agent/          # Workflow-based agents
│   ├── mape_k_approach/         # MAPE-K approach (4 components)
│   ├── mape_k_2_components/     # MAPE-K approach (2 components)
│   ├── single_agent_approach/   # Single agent approach
│   ├── o1_agent/                # O1 agent implementation
│   ├── knowledge_base/          # XAI methods knowledge base
│   ├── utils/                   # Utility functions
│   ├── base_agent.py            # Base agent class
│   ├── xai_utils.py             # XAI utility functions
│   ├── xai_prompts.py           # Prompt templates
│   ├── output_parsers.py        # Output parsers
│   ├── query_engine_tools.py    # Knowledge base query tools
│   ├── pg_vec_store.py          # Vector database helpers
│   └── explanation_state.py     # Explanation states
├── docker-compose.yml           # Docker configuration
└── Dockerfile                   # Docker image definition
```

## Core Components

### Base Agent Structure

All XAI agents implement the `XAIBaseAgent` abstract class defined in `base_agent.py`. This class provides a common interface for different agent implementations with the main method:

```python
async def answer_user_question(self, user_question):
    """
    Answer the user's question based on the initialized data point.
    Returns a tuple: (analysis, response, recommend_visualization)
    """
```

### Workflow Agents

Located in the `workflow_agent/` directory, these agents implement different workflows to answer user questions:

1. **XAIWorkflowAgent** (`workflow_agent.py`): Implements a step-by-step workflow:
   - Receives user messages
   - Analyzes questions
   - Selects appropriate XAI methods
   - Prepares responses
   - Calls LLM to generate responses
   - Evaluates and processes responses

2. **SimpleXAIWorkflowAgent** (`simple_workflow_agent.py`): Implements a simpler workflow:
   - Receives user messages
   - Directly analyzes and answers questions

### MAPE-K Approach

The MAPE-K (Monitor-Analyze-Plan-Execute-Knowledge) approach is implemented in two different ways:

1. **MapeKXAIWorkflowAgent** (`mape_k_approach/mape_k_workflow_agent.py`): Full MAPE-K loop with 4 components:
   - **Monitor**: Analyzes user messages to determine understanding level
   - **Analyze**: Updates the user model
   - **Plan**: Creates or updates explanation plans
   - **Execute**: Generates responses to users

2. **MapeK2Component** (`mape_k_2_components/mape_k_workflow_agent.py`): Simplified MAPE-K with 2 components:
   - **Monitor**: Analyzes user messages
   - **Scaffolding**: Plans and implements explanation strategies

### Knowledge Base

The `knowledge_base/` directory contains:
- Markdown files with information about XAI methods (e.g., `CeterisParibus.md`, `Anchors.md`)
- `create_index.py`: Processes markdown files and adds them to the vector database

### Vector Database

The system uses PostgreSQL with the pgvector extension to store vectorized information about XAI methods:
- `docker-compose.yml`: Docker configuration
- `Dockerfile`: Docker image definition
- `init_pgvector.sql`: Database schema
- `pg_vec_store.py`: Helper functions for database interaction

### Utility Tools

- `xai_utils.py`: Functions for processing XAI explanations
- `xai_prompts.py`: Prompt templates for LLM interactions
- `output_parsers.py`: Parsers for processing LLM responses
- `query_engine_tools.py`: Tools for querying the knowledge base

## Data Flow

1. **User Question**: User asks about an ML model prediction
2. **Agent Processing**: 
   - The question is analyzed
   - Appropriate XAI methods are selected
   - A response is prepared using the selected methods
   - LLM generates the final response
3. **Response Delivery**: The response is sent to the user

In the MAPE-K approach, the flow is more adaptive:
1. **Monitor**: Analyze user understanding level
2. **Analyze**: Update user model
3. **Plan**: Create/update explanation plan
4. **Execute**: Generate response
5. **Knowledge**: Retrieve relevant XAI information

## Key Functions

### XAIWorkflowAgent
- `analyze_question`: Selects appropriate XAI methods
- `prepare_response`: Prepares prompts for LLM
- `call_llm_agent`: Calls LLM to generate responses

### MapeKXAIWorkflowAgent
- `monitor`: Determines user understanding level
- `analyze`: Updates user model
- `plan`: Creates explanation plans
- `execute`: Generates responses

### Utility Functions
- `process_xai_explanations`: Processes XAI explanations
- `get_citation_query_engine`: Creates knowledge base query engines
- `build_query_engine_tools`: Creates query tools for LLM

## Explanation States

The system tracks the state of explanations using the `ExplanationState` enum:
```python
class ExplanationState(Enum):
    NOT_YET_EXPLAINED = "not_yet_explained"
    SHOWN = "shown"
    UNDERSTOOD = "understood"
    NOT_UNDERSTOOD = "not_understood"
    PARTIALLY_UNDERSTOOD = "partially_understood"
```

## Setup and Usage

### Requirements
- Python 3.8+
- PostgreSQL with pgvector extension
- Docker (optional)

### Basic Setup
1. Clone the repository
2. Install dependencies
3. Configure environment variables
4. Start the vector database
5. Build the knowledge base
6. Start the API and frontend

## Development

### Adding a New Agent
1. Create a class that implements `XAIBaseAgent`
2. Implement the `answer_user_question` method
3. Add the agent to the API

### Adding a New XAI Method
1. Create a markdown file in the knowledge base
2. Rebuild the knowledge base index 