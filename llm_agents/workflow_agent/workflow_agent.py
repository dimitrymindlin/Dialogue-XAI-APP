import json
from typing import List, Union
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.prompts import display_prompt_dict
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
from llm_agents.xai_prompts import get_analysis_prompt_template, get_response_prompt_template
from llm_agents.output_parsers import get_analysis_output_parser, get_response_output_parser


# Define custom events

class PrepEvent(Event):
    pass


class JudgeEvent(Event):
    response: str


class AnalysisEvent(Event):
    analysis: str
    selected_methods: List[str]


class InputEvent(Event):
    input: List[ChatMessage]


class ResponseEvent(Event):
    response: ChatMessage


# Define the XAIAgent class using the Workflow paradigm
class XAIWorkflowAgent(Workflow, XAIBaseAgent):
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
        self.memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
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
    def initialize_new_datapoint(self, instance_information, xai_explanations, predicted_class_name):
        self.xai_explanations = process_xai_explanations(xai_explanations)
        self.instance = extract_instance_information(instance_information)
        self.predicted_class_name = predicted_class_name
        self.memory = ChatMemoryBuffer.from_defaults(llm=self.llm)  # Reset chat history

    # Method to get chat history as text
    def get_chat_history_text(self):
        chat_history_messages = self.memory.get()
        chat_history_text = ""
        for msg in chat_history_messages:
            if msg.role == "user":
                chat_history_text += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                chat_history_text += f"Assistant: {msg.content}\n"
        return chat_history_text

    # Step to handle new user message
    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # Get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)
        self.agent.memory.put(user_msg)
        # Clear current reasoning
        await ctx.set("current_reasoning", [])
        return PrepEvent()

    # Step to analyze the user's question and select xAI methods
    @step
    async def analyze_question(self, ctx: Context, ev: PrepEvent) -> AnalysisEvent:
        # Get chat history text
        chat_history_text = self.get_chat_history_text()

        # Prepare the analysis prompt
        analysis_prompt = get_analysis_prompt_template().format(
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
        chat_history_messages = self.memory.get()
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
        try:
            response_dict = self.analysis_output_parser.parse(response_content)
            analysis = response_dict['analysis']
            selected_methods = response_dict['selected_methods']
        except Exception as e:
            analysis = "Unable to parse analysis."
            selected_methods = []

        # Store the analysis in context
        await ctx.set("analysis", analysis)
        await ctx.set("selected_methods", selected_methods)

        print("Analysis: ", analysis)
        print("Selected Methods: ", selected_methods)

        return AnalysisEvent(analysis=analysis, selected_methods=selected_methods)

    # Step to prepare the final response
    @step
    async def prepare_response(self, ctx: Context, ev: AnalysisEvent) -> InputEvent:
        # Get chat history text
        chat_history_text = self.get_chat_history_text()

        # Retrieve selected explanations
        selected_methods = ev.selected_methods
        # Get the explanations for the selected methods
        selected_explanations = {method: self.xai_explanations.get(method, {}) for method in selected_methods}

        # Prepare the response prompt
        response_prompt = get_response_prompt_template().format(
            domain_description=self.domain_description,
            feature_names=self.feature_names,
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            selected_explanations=selected_explanations,
            chat_history=chat_history_text,
        )

        # Include format instructions
        response_prompt += "\n\n" + self.response_format_instructions

        # Get the last user message
        chat_history_messages = self.agent.memory.get()
        if chat_history_messages and chat_history_messages[-1].role == "user":
            user_message = chat_history_messages[-1]
        else:
            user_message = ChatMessage(role="user", content="")

        messages = [
            ChatMessage(role="system", content=response_prompt),
            user_message
        ]

        print("Response Prompt: ", response_prompt)

        return InputEvent(input=messages)

    # Step to call the LLM for the final response
    @step
    async def call_llm_agent(self, ctx: Context, ev: InputEvent) -> ResponseEvent:
        messages = ev.input
        # Create agent with system prompt
        self.agent = OpenAIAgent.from_tools(
            tools=build_query_engine_tools(),
            llm=self.llm,
            verbose=True,
            system_prompt=messages[0].content,
        )
        response = self.agent.chat(messages[1].content)
        self.memory = self.agent.memory
        return ResponseEvent(response=ChatMessage(role="assistant", content=response.response))

    @step
    async def judge_response(self, ctx: Context, ev: JudgeEvent) -> Union[InputEvent, StopEvent]:
        # Retrieve the user's question and the assistant's response
        assistant_response = ev.response
        chat_history_messages = self.memory.get()
        user_question = ""
        for msg in reversed(chat_history_messages):
            if msg.role == "user":
                user_question = msg.content
                break

        # Prepare the judging prompt
        judging_prompt = f"""
    You are an AI assistant tasked with evaluating the following response for clarity and completeness:

    <<User Question>>:
    {user_question}

    <<Assistant's Response>>:
    {assistant_response}

    <<Task>>:
    Determine if the assistant's response is clear and self-contained in answering the user's question. If it is sufficient, reply with true. If not, provide specific feedback on what is missing or unclear.

    <<Response Instructions>>:
    Return a JSON object with the following keys:
    - "is_sufficient": true or false
    - "feedback": "Your feedback if the response is not sufficient."
    """

        messages = [ChatMessage(role="system", content=judging_prompt)]

        # Call the LLM to judge the response
        try:
            response = await self.llm.achat(messages)
        except Exception as e:
            print(f"Error during LLM call: {e}")

        # Parse the response
        try:
            response_dict = json.loads(response.message.content)
            is_sufficient = response_dict.get("is_sufficient", False)
            feedback = response_dict.get("feedback", "")
        except Exception as e:
            is_sufficient = False
            feedback = "Unable to parse the judgment response."

        if is_sufficient:
            # Response is sufficient, proceed to stop the workflow
            return StopEvent(result={
                "analysis": await ctx.get("analysis", ""),
                "response": assistant_response,
                "recommend_visualization": await ctx.get("recommend_visualization", False)
            })
        else:
            # Response needs refinement, prepare a new prompt to refine the answer
            refinement_prompt = f"""
    You are an AI assistant. The previous response to the user's question was not clear or self-contained.

    <<User Question>>:
    {user_question}

    <<Feedback on Previous Response>>:
    {feedback}

    <<Task>>:
    Refine the previous response to address the feedback and ensure it is clear and self-contained.

    <<Response Instructions>>:
    Provide a revised response to the user's question.
    """

            messages = [
                ChatMessage(role="system", content=refinement_prompt),
            ]

            # Return InputEvent to call the LLM again for refinement
            return InputEvent(input=messages)

    # Step to parse the LLM's response
    @step
    async def parse_response(self, ctx: Context, ev: ResponseEvent) -> JudgeEvent:
        # Parsing logic
        response_content = ev.response.content
        # Parse the response using the output parser
        try:
            response_dict = self.response_output_parser.parse(response_content)
            response = response_dict['response']
            recommend_visualization = response_dict.get('recommend_visualization', False)
        except Exception as e:
            # Handle parsing error
            response = response_content  # Return the raw response
            recommend_visualization = False
        # Add the assistant's response to memory
        assistant_message = ChatMessage(role="assistant", content=response)
        self.memory.put(assistant_message)
        self.agent.memory.put(assistant_message)

        # Store recommendation in context
        await ctx.set("recommend_visualization", recommend_visualization)

        # Return JudgeEvent to proceed to the judging step
        return JudgeEvent(response=response)

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
    draw_all_possible_flows(XAIWorkflowAgent, filename="xaiagent_flow_all.html")
