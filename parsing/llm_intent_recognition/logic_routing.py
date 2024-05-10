from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    method: Literal["feature_importances", "counterfactual"] = Field(
        ...,
        description="Given a user question choose which XAI method would be most suitable for answering their question",
    )


# LLM with function call
llm = ChatOllama(model="llama3", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to the appropriate xai method.

Based on the underlying information intent the question is referring to, route it to the relevant xai method."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router
router = prompt | structured_llm

question = """What is the most important feature?"""

result = router.invoke({"question": question})
print(result)