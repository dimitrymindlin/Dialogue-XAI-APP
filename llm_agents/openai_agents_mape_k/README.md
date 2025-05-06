# OpenAI Agents MAPE-K Implementation

This directory contains an implementation of the MAPE-K (Monitor-Analyze-Plan-Execute over Knowledge) agent using OpenAI's Agents SDK, replacing the original llama_index-based implementation.

## Overview

This implementation provides a comprehensive architecture with three main components:

1. **Decision Agent** - Routes user queries to either the MAPE-K agent or RAG system based on query type
2. **MAPE-K Agent** - Handles model explanation queries using the unified prompt from `merged_prompts.py`
3. **RAG System** - Retrieves information from CSV and PDF documents to answer dataset/domain questions

The architecture follows this flow:
1. User query is received by the DecisionAgent
2. DecisionAgent determines if the query is about model explanations or dataset information
3. Query is routed to the appropriate system (MAPE-K agent or RAG system)
4. Response is generated and returned to the user

## Key Components

- `decision_agent.py` - Entry point that routes queries based on type
- `mape_k_agent.py` - Main MAPE-K implementation using the unified prompt
- `simple_agent.py` - Simplified MAPE-K implementation (legacy)
- `rag_system.py` - RAG system for retrieving information from documents
- `__init__.py` - Exports the agent classes
- Test scripts for each component

## Usage

### 1. Using the Decision Agent (recommended)

This is the main entry point that will route queries appropriately:

```python
from llm_agents.openai_agents_mape_k import DecisionAgent

# Initialize the agent
agent = DecisionAgent(
    feature_names="age, education, occupation, etc.",
    domain_description="An income prediction model that determines if a person earns above or below $50K",
    user_ml_knowledge="beginner",
    experiment_id="user_123"
)

# Answer a user question
reasoning, response = await agent.answer_user_question("Why was this person predicted to earn more than $50K?")
```

### 2. Using Only the MAPE-K Agent

If you only need the MAPE-K functionality with the unified prompt:

```python
from llm_agents.openai_agents_mape_k import MapeKAgent

# Initialize the agent
agent = MapeKAgent(
    feature_names="age, education, occupation, etc.",
    domain_description="An income prediction model",
    user_ml_knowledge="beginner",
    experiment_id="user_123"
)

# Initialize a datapoint
agent.initialize_new_datapoint(
    instance=instance_datapoint, 
    xai_explanations=explanations,
    xai_visual_explanations=visual_explanations,
    predicted_class_name="Approved",
    opposite_class_name="Rejected",
    datapoint_count=0
)

# Answer a user question
reasoning, response = await agent.answer_user_question("How does education affect the prediction?")
```

### 3. Using the RAG System Directly

If you only need the RAG functionality:

```python
from llm_agents.openai_agents_mape_k.rag_system import RAGSystem

# Initialize the RAG system
rag_system = RAGSystem(domain_description="Income prediction model dataset")

# Answer a question about the dataset
reasoning, response = await rag_system.answer_query("What is the average age in the dataset?")
```

## Integration with ExplainBot

To use the new architecture with ExplainBot:

1. **Decision Agent (recommended)**:
   ```python
   bot = ExplainBot(
       study_group="interactive",
       ml_knowledge="beginner",
       user_id="user_123",
       use_llm_agent="decision_agent"
   )
   ```

2. **MAPE-K Agent Only**:
   ```python
   bot = ExplainBot(
       study_group="interactive",
       ml_knowledge="beginner",
       user_id="user_123",
       use_llm_agent="mape_k_agent"
   )
   ```

3. **Simple Agent (legacy)**:
   ```python
   bot = ExplainBot(
       study_group="interactive",
       ml_knowledge="beginner",
       user_id="user_123",
       use_llm_agent="openai_agents"
   )
   ```

## Document Indexing for RAG

The RAG system can index CSV and PDF files from the `data/documents` directory:

1. Create the directory if it doesn't exist:
   ```
   mkdir -p data/documents
   ```

2. Place your CSV and PDF files in the directory.

3. Run the RAG system, which will automatically index the files:
   ```python
   from llm_agents.openai_agents_mape_k.rag_system import RAGSystem, index_documents
   
   # Force reindexing if needed
   await index_documents(force_reindex=True)
   
   # Or let the RAG system handle indexing automatically
   rag_system = RAGSystem(domain_description="Your domain")
   await rag_system.answer_query("Your question")
   ```

## Testing the Architecture

To test the complete architecture:

```bash
python llm_agents/openai_agents_mape_k/test_decision_architecture.py
```

This will:
1. Create a test dataset
2. Initialize the vector store
3. Test routing model explanation queries to the MAPE-K agent
4. Test routing dataset queries to the RAG system

## Requirements

- Python 3.8+
- OpenAI API key (set in .env file or environment variable)
- agents package (OpenAI Agents SDK)
- PyPDF2 (for PDF processing, optional)
- pandas, numpy, etc. (see requirements.txt) 