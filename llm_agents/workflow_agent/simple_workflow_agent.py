from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.workflow import (
    Event,
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.llms.openai import OpenAI

from llm_agents.base_agent import XAIBaseAgent
from llm_agents.query_engine_tools import build_query_engine_tools

from llm_agents.xai_utils import process_xai_explanations, extract_instance_information
from llm_agents.xai_prompts import get_single_answer_prompt_template, get_augment_user_question_prompt_template
from llm_agents.output_parsers import get_analysis_output_parser, get_response_output_parser


# Define custom events

class PrepEvent(Event):
    pass


# Define the XAIAgent class using the Workflow paradigm
class SimpleXAIWorkflowAgent(Workflow, XAIBaseAgent):
    def __init__(
            self,
            llm: LLM = None,
            feature_names="",
            domain_description="",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.xai_explanations = None
        self.predicted_class_name = None
        self.instance = None
        self.llm = llm or OpenAI()
        self.agent = OpenAIAgent.from_tools(
            tools=build_query_engine_tools(),
            llm=self.llm,
            verbose=True,
            system_prompt="You are an AI assistant specialized in answering queries about research documents.",
        )

        # Initialize the output parsers
        self.analysis_output_parser = get_analysis_output_parser()
        self.analysis_format_instructions = self.analysis_output_parser.get_format_instructions()

        self.response_output_parser = get_response_output_parser()
        self.response_format_instructions = self.response_output_parser.get_format_instructions()

    # Method to initialize a new datapoint
    def initialize_new_datapoint(self,
                                 instance_information,
                                 xai_explanations,
                                 predicted_class_name,
                                 opposite_class_name):
        self.xai_explanations = process_xai_explanations(xai_explanations, predicted_class_name, opposite_class_name)
        self.instance = extract_instance_information(instance_information)
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.agent.memory.reset()

    # Step to handle new user message
    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # Get user input
        user_input = ev.input
        # Augment user message by history to create stand alone message
        chat_history_text = [(message.role, message.content) for message in self.agent.memory.get_all()]
        if len(chat_history_text) > 0:
            augment_prompt = PromptTemplate(get_augment_user_question_prompt_template().format(
                chat_history="\n".join([f"{role}: {content}" for role, content in chat_history_text]),
                new_user_input=user_input))
            augmented_user_input = await self.llm.apredict(augment_prompt)
            print(f"Augmented user input: {augmented_user_input}")
        else:
            augmented_user_input = user_input

        user_msg = ChatMessage(role="user", content=augmented_user_input)
        self.agent.memory.put(user_msg)
        # Clear current reasoning
        await ctx.set("current_reasoning", [])
        return PrepEvent()

    # Step to analyze the user's question and select xAI methods
    @step
    async def answer_question(self, ctx: Context, ev: PrepEvent) -> StopEvent:
        # Get chat history text
        chat_history_text = [(message.role, message.content) for message in self.agent.memory.get_all()[:-1]]

        # Prepare the analysis prompt
        analysis_prompt = get_single_answer_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            xai_explanations=self.xai_explanations,
            chat_history=chat_history_text,
        )

        # Include format instructions
        analysis_prompt += "\n\n" + self.analysis_format_instructions

        # Get the last user message
        chat_history_messages = self.agent.memory.get()
        if chat_history_messages and chat_history_messages[-1].role == "user":
            user_message = chat_history_messages[-1]
        else:
            user_message = ChatMessage(role="user", content="")

        messages = [
            ChatMessage(role="system", content=analysis_prompt),
            user_message
        ]

        # Call the LLM for analysis
        response = await self.llm.achat(messages)

        # Parse the response
        response_content = response.message.content

        # Save response to history
        response_msg = ChatMessage(role="system", content=response_content)
        self.agent.memory.put(response_msg)

        try:
            response_dict = self.analysis_output_parser.parse(response_content)
            analysis = response_dict['analysis']
            selected_methods = response_dict['selected_methods']
            response_string = response_dict['response']
        except Exception as e:
            response_dict = self.analysis_output_parser.parse(response_content)
            response_string = response_dict['analysis']
            analysis = ""
            selected_methods = ""

        # Store the analysis in context
        await ctx.set("analysis", analysis)
        await ctx.set("selected_methods", selected_methods)

        return StopEvent(result={
            "analysis": await ctx.get("analysis", ""),
            "selected_methods": await ctx.get("selected_methods", []),
            "response": response_string,
        })

    # Method to answer user question
    async def answer_user_question(self, user_question):
        ret = await self.run(input=user_question)
        analysis = ret.get("analysis", "")
        response = ret.get("response", "")
        return analysis, response


from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

if __name__ == "__main__":
    # Draw all
    draw_all_possible_flows(SimpleXAIWorkflowAgent, filename="simple_xaiagent_flow_all.html")
