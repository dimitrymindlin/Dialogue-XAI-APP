# Import things that are needed generically
import os

from langchain_community.embeddings import OllamaEmbeddings

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"

from typing import Optional, Type, Any, Literal

from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.tools import tool
from langchain.utils.math import cosine_similarity
from langchain.tools.render import render_text_description_and_args
from langchain_core.runnables import RunnableLambda, RunnablePassthrough



@tool
def feature_importance(feature: str) -> str:
    """Calculate the importance of a specific feature."""
    return f"Calculating importance for {feature}..."


@tool
def prediction_change(feature: str, change: float) -> str:
    """Determine the effect of changing a feature on predictions."""
    return f"Predicting changes for {feature} by {change} units..."


@tool
def feature_effect(feature: str) -> str:
    """Assess the effect of a specific feature on the model."""
    return f"Assessing effect of {feature} on the model..."


def call_tools(msg):
    """Sequential tool calling helper that executes tools based on the AI message."""
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


system_prompt = """You are an assistant to answer user queries about machine learning model decisions for 
prediction the income of individuals.
Answer the following questions as best you can. You have access to the following tools:

{tools}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding."""

# Example setup of the DecisionLLM
llm = OllamaFunctions(model="llama2:7b")  # Initialize the LLM with appropriate model
tools = [feature_importance, prediction_change, feature_effect]  # Create a list of tools
llm_with_tools = llm.bind_tools(tools=tools)  # Bind the tools to the LLM
tool_map = {tool.name: tool for tool in tools}  # Create a map of tool names to tool objects

chain = llm_with_tools | call_tools

# Test cases
print(chain.invoke("What is the importance of the age feature in the model?"))
print(chain.invoke("How does increasing the price by 10% affect predictions?"))
print(chain.invoke("What effect does the location feature have on the model?"))
