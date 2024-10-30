from abc import ABC, abstractmethod

from llama_index.core.workflow.workflow import WorkflowMeta


class XAIBaseAgentMeta(WorkflowMeta, type(ABC)):
    pass


class XAIBaseAgent(ABC, metaclass=XAIBaseAgentMeta):
    @abstractmethod
    async def answer_user_question(self, user_question):
        """
        Answer the user's question based on the initialized data point.
        Returns a tuple: (analysis, response, recommend_visualization)
        """
        pass
