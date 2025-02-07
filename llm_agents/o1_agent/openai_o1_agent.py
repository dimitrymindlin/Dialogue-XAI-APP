import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.utils.postprocess_message import replace_plot_placeholders


def get_system_propmpt(explanation_collection, instance, prediction, previous_knowledge):
    return f"""
<<Role>>

You are a tutor agent focused on explaining and discussing AI model predictions in a way that is accurate, concise, and tailored to the user’s evolving understanding. You must track internally what you have explained, how the user reacted, and any signs of misunderstanding or deeper interest. Do not talk about other things than the prediction and the explanation of the prediction. If the user asks things not related to your task, try to bring the conversation back to the prediction and the explanation of the prediction.

<<Context>>
- Domain Description: Adult Census Income Dataset, commonly used for binary classification tasks in machine learning. The goal is to predict whether an individual’s income exceeds $50K/year based on demographic and employment-related features. It is based on US Census data and is often used to explore income distribution patterns, discrimination, or bias in predictive models.
- Model Features: ['Age', 'Education Level', 'Marital Status', 'Occupation', 'Weekly Working Hours', 'Work Life Balance', 'Investment Outcome']
- Current local Instance of interest: {instance}
- Predicted Class by AI Model: {prediction}

<<Task>>
	1.	Monitor the User understanding
	•		Maintain accuracy (Quality) and relevance (Relation) by focusing on the user’s immediate needs.
	2.	Assess and Adapt
	•		Estimate the user’s current knowledge or misconceptions from prior exchanges.
	•		Decide how much detail is sufficient (Quantity) and present it clearly (Manner). Use HTML formatting for clarity and emphasis.
	•		Encourage the user’s engagement with prompts or follow-up questions—if the user seems passive, nudge them toward more interactive or constructive participation.
	3.	Explain and Iterate
	•		Provide fact-based, context-appropriate explanations, using short paragraphs or structured bullet points.
	•		Invite clarification questions to support the user’s understanding. Remain flexible to re-explain or simplify if confusion persists. Expect to iterate on explanations as needed, instead of providing a one-time, exhaustive answer.
	4.	Foster Co-Construction
	•		Align with the ICAP framework: encourage the user to restate or elaborate on your explanations, reinforcing constructive or interactive engagement.
	•		Adjust depth and complexity based on how the user’s responses evolve, integrating incremental or more advanced explanations where relevant.

Your goal is to deliver accurate, concise, and user-centered explanations that build upon each user interaction, while internally tracking conversational context and the user’s evolving comprehension. Make sure that the user knows the concepts before you talk about feature importances or counterfactual explanations. The user might not be familiar with the way Machine Learning works and what features are. Act like a professional teacher, eliciting the information and scaffolding the user in his understanding of the prediction.

<<Explanation Collection>>
{explanation_collection}

<<User knowledge after previous conversations>>
You can assume that the user knows the following concepts, if not empty:
{previous_knowledge}
"""


"""load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

assistant = client.beta.assistants.create(
    name="XAI Tutor",
    instructions=system_prompt,
    model="o1",
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Can you explain how the model prediction was made?"
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

while True:
    current_run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    if current_run.status == "completed":
        break
    time.sleep(1)

messages = client.beta.threads.messages.list(
    thread_id=thread.id,
)

for message in reversed(messages.data):
    print(f"{message.role}: {message.content[0].text.value}")"""


class XAITutorAssistant:
    def __init__(self, feature_names, domain_description, user_ml_knowledge, verbose):
        """
        #TODO Include the domain description and the feature names in the prompt
        """
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.assistant = None
        self.thread = self.client.beta.threads.create()
        self.user_knowlede_summary = None
        self.visual_explanations_dict = None
        self.data_point_counter = 0

    async def persist_user_knowledge(self):
        """
        Condense the main concepts the user learned to not explain them again for the next datapoint
        """
        persistance_prompt = "Given what we discussed, summarize my level of understanding and the concepts I understand" \
                             "so we can discuss a new datapoint and model prediction and you can build on my knowledge." \
                             "While the explanation values will change, I will not need to know the same basics again." \
                             "If I know nothing, you can leave this empty. Otherwise, write a small summary with my " \
                             "understanding of the concepts and engagement level and any preferences such as seeing plots" \
                             "or not wanting or understanding plots."
        _, summary = await self.answer_user_question(persistance_prompt)
        self.user_knowlede_summary = summary
        return summary

    async def initialize_new_datapoint(self,
                                 instance: InstanceDatapoint,
                                 xai_explanations,
                                 xai_visual_explanations,
                                 predicted_class_name,
                                 opposite_class_name):
        # If the user_model is not empty, store understood and not understood concept information in the user model
        # and reset the rest to not_explained
        if self.data_point_counter > 0:
            previous_knowledge = await self.persist_user_knowledge()
        else:
            previous_knowledge = ""

        instance = instance.displayable_features
        self.visual_explanations_dict = xai_visual_explanations
        # Set user model
        self.populator = XAIExplanationPopulator(
            template_dir=".",
            template_file="llm_agents/mape_k_approach/plan_component/explanations_model.yaml",
            xai_explanations=xai_explanations,
            predicted_class_name=predicted_class_name,
            opposite_class_name=opposite_class_name,
            instance_dict=instance
        )
        # Populate the YAML
        self.populator.populate_yaml()
        # Validate substitutions
        self.populator.validate_substitutions()
        # Optionally, retrieve as a dictionary
        populated_yaml_dict = self.populator.get_populated_yaml(as_dict=True)
        system_prompt = get_system_propmpt(populated_yaml_dict, instance, predicted_class_name, previous_knowledge)
        self.assistant = self.client.beta.assistants.create(
            name="XAI Tutor",
            instructions=system_prompt,
            model="o1",
        )
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=f"The model predicts that the current Person is {predicted_class_name}. If you have questions about the prediction feel free to ask me."
        )

    async def answer_user_question(self, user_question):

        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_question
        )

        # Create a run to trigger the assistant's response.
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )

        # Poll until the assistant's response run is complete.
        while True:
            current_run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            if current_run.status == "completed":
                break
            await asyncio.sleep(1)

        # Retrieve all messages in the thread.
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)

        # Extract the latest assistant response.
        assistant_message = None
        for message in reversed(messages.data):
            if message.role == "assistant":
                assistant_message = message.content[0].text.value

        # Postprocess the assistant's response.
        assistant_message = replace_plot_placeholders(assistant_message, self.visual_explanations_dict)

        return "", assistant_message
