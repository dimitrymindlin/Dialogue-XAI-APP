import json
import logging
import os

import pytest
from llama_index.core import PromptTemplate

from llm_agents.mape_k_approach.execute_component.execute_prompt import get_execute_prompt_template
from llm_agents.mape_k_approach.analyze_component.analyze_prompt import get_analyze_prompt_template, AnalyzeResult
from llm_agents.mape_k_approach.plan_component.advanced_plan_prompt_multi_step import PlanResultModel, \
    get_plan_prompt_template, ChosenExplanationModel
from llm_agents.mape_k_approach.monitor_component.definition_wrapper import ICAPModes
from llm_agents.mape_k_approach.monitor_component.monitor_prompt import get_monitor_prompt_template, MonitorResultModel
from llm_agents.mape_k_approach.mape_k_workflow_agent import AugmentResult, \
    ExecuteResult
from llm_agents.mape_k_approach.monitor_component.understanding_displays import DefinitionWrapper
from llm_agents.mape_k_approach.user_model_fine_grained import UserModelFineGrained, ExplanationState
from llm_agents.mape_k_approach.plan_component.xai_exp_populator import XAIExplanationPopulator
from llm_agents.xai_prompts import get_augment_user_question_prompt_template
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')
LLM_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust to DEBUG/INFO as needed

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Adjust level if needed

# Define log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

llm = OpenAI(model=LLM_MODEL_NAME)

# Create Test Instance
instance = {'Age': '23',
            'Education Level': 'Middle School',
            'Investment Outcome': 'No Investment',
            'Marital Status': 'Married',
            'Occupation': 'Admin',
            'Weekly Working Hours': '40',
            'Work Life Balance': 'Fair'}

domain_description = "Adult Census Income Dataset, commonly used for binary classification tasks in machine learning. " \
                     "The goal is to predict whether an individual’s income exceeds $50K/year based on demographic and " \
                     "employment-related features. It is based on US Census data and is often used to explore income " \
                     "distribution patterns, discrimination, or bias in predictive models."
feature_names = ['Age', 'EducationLevel', 'MaritalStatus', 'Occupation', 'WeeklyWorkingHours', 'WorkLifeBalance',
                 'InvestmentOutcome']
predicted_class_name = 'under 50k'
opposite_class_name = 'over 50k'
understanding_displays = DefinitionWrapper("monitor_component/understanding_displays_definition.json")
engagement_modes = ICAPModes("monitor_component/icap_modes_definition.json")

xai_explanations = {'model_prediction': 'under 50k', 'instance_type': 'Person', 'feature_importance': (
    {'MaritalStatus': [-0.22929119], 'InvestmentOutcome': [0.17052516], 'Age': [0.11885647],
     'WeeklyWorkingHours': [0.059259847], 'EducationLevel': [0.058852844], 'Occupation': [-0.012996635],
     'WorkLifeBalance': [-0.0019427723]}, 1), 'opposite_class': 'over 50k',
                    'counterfactuals': '1. Changing <b>Work Life Balance</b> to <b>Good</b> and Changing <b>Investment Outcome</b> to <b>Major Gain (above 5k$)</b><br>2. Decreasing <b>Weekly Working Hours</b> to <b>34</b> and Changing <b>Investment Outcome</b> to <b>Major Gain (above 5k$)</b><br>3. Changing <b>Investment Outcome</b> to <b>Major Gain (above 5k$)</b><br> <br>There are other possible changes. These are just examples.',
                    'anchors': 'Keeping these conditions: <br><b>Age is not above 28</b> and <b>Occupation = Admin</b> and <b>Investment Outcome = No Investment</b><br>the prediction will most likely stay the same.',
                    'feature_statistics': {
                        'Age': '<b>Age</b> ranges from 17 years to 90 years with a mean of 38,42 years.',
                        'Marital Status': 'The value <b>Single</b> occurs <b>11175</b> times.<br>The value <b>Married</b> occurs <b>10075</b> times.<br>',
                        'Occupation': 'The value <b>Blue-Collar</b> occurs <b>7732</b> times.<br>The value <b>Professional</b> occurs <b>2684</b> times.<br>The value <b>White-Collar</b> occurs <b>2668</b> times.<br>The value <b>Admin</b> occurs <b>2530</b> times.<br>The value <b>Sales</b> occurs <b>2389</b> times.<br>The value <b>Service</b> occurs <b>2210</b> times.<br>The value <b>Other</b> occurs <b>1032</b> times.<br>The value <b>Military</b> occurs <b>5</b> times.<br>',
                        'Weekly Working Hours': '<b>Weekly Working Hours</b> ranges from 16 to 65 with a mean of 40,78.',
                        'Work Life Balance': 'The value <b>Fair</b> occurs <b>10531</b> times.<br>The value <b>Good</b> occurs <b>5386</b> times.<br>The value <b>Poor</b> occurs <b>5333</b> times.<br>',
                        'Education Level': "The value <b>Middle School</b> occurs <b>11645</b> times.<br>The value <b>High School Graduate</b> occurs <b>3524</b> times.<br>The value <b>Primary Education</b> occurs <b>2640</b> times.<br>The value <b>High School without Graduation</b> occurs <b>1678</b> times.<br>The value <b>College without Degree</b> occurs <b>1137</b> times.<br>The value <b>Associate's Degrees</b> occurs <b>362</b> times.<br>The value <b>Bachelor's Degree</b> occurs <b>264</b> times.<br>",
                        'Investment Outcome': 'The value <b>No Investment</b> occurs <b>18486</b> times.<br>The value <b>Major Gain (above 5k$)</b> occurs <b>1085</b> times.<br>The value <b>Major Loss (more than 1k$)</b> occurs <b>985</b> times.<br>The value <b>Minor Gain (up to 5k$)</b> occurs <b>676</b> times.<br>The value <b>Minor Loss (up to 1k$)</b> occurs <b>18</b> times.<br>'},
                    'ceteris_paribus': ['No changes in <b>Age</b> alone can change the model prediction.',
                                        'No changes in <b>Marital Status</b> alone can change the model prediction.',
                                        'No changes in <b>Occupation</b> alone can change the model prediction.',
                                        'No changes in <b>Weekly Working Hours</b> alone can change the model prediction.',
                                        'No changes in <b>Work Life Balance</b> alone can change the model prediction.',
                                        'No changes in <b>Education Level</b> alone can change the model prediction.',
                                        'Changing <b>Investment Outcome</b> to <br> <b>Major Gain (above 5k$)</b>  will switch the prediction to <b>over 50k.']}

populator = XAIExplanationPopulator(
    template_dir=".",
    template_file="plan_component/explanations_model.yaml",
    xai_explanations=xai_explanations,
    predicted_class_name=predicted_class_name,
    opposite_class_name=opposite_class_name,
    instance_dict=instance
)

# Populate the YAML
populator.populate_yaml()

# Validate substitutions
populator.validate_substitutions()

# Retrieve the populated YAML as a string
populated_yaml_str = populator.get_populated_yaml()

# Optionally, retrieve as a dictionary
populated_yaml_dict = populator.get_populated_yaml(as_dict=True)

user_model = UserModelFineGrained()
user_model.set_model_from_summary(populated_yaml_dict)

history_after_scaffolding = ["User: What is the most important feature?",
                             "Agent: The most important feature is Marital Status with an importance of 35 percent. "
                             "This attribute is in favour of the opposite prediction, which is over 50k.",
                             "User: If the most important feature is in favour of the other class, why is it still under 50k?",
                             "Agent: That's a great question! The model's prediction of under 50k can be influenced by multiple"
                             " features working together. While Marital Status is important and leans towards predicting "
                             "over 50k, other features like Age, Education Level, and Occupation also play significant "
                             "roles. In this case, the combination of these features may lead to a prediction of under "
                             "50k. Can you summarize how you think these features might interact to affect the prediction?"
                             "User: Ah okay, so because the person is young and has a low education level, the model "
                             "predicts under 50k income?"]

"""previous_explanations = [{'FeatureImportances': 'Concept'},
                         {'FeatureImportances': 'FeaturesInFavourOfUnder50k'},
                         {'Counterfactuals': 'Concept'},
                         {'Counterfactuals': 'ImpactMultipleFeatures'}]

for exp in previous_explanations:
    exp_name = list(exp.keys())[0]
    exp_step = exp[exp_name]
    user_model.mark_explanation_step_as_understood(exp_name, exp_step)"""


@pytest.mark.skip
def test_augment_prompt():
    chat_history = ""
    user_input = "Why did the model predict this class?"

    augment_prompt = get_augment_user_question_prompt_template().format(
        chat_history=chat_history,
        new_user_input=user_input)
    augment_prompt = PromptTemplate(augment_prompt)
    augmented_user_input = llm.structured_predict(AugmentResult, augment_prompt)

    logger.info(f"Augmented user input: {augmented_user_input.new_user_input}")
    assert isinstance(augmented_user_input.reasoning, str)
    assert isinstance(augmented_user_input.new_user_input, str)


@pytest.mark.parametrize(
    "chat_history, user_message, expected_displays, expected_engagement",
    [

        # Case Identifying signal understanding after scaffolding
        (
                history_after_scaffolding[:-1],
                history_after_scaffolding[-1],
                ["signal_understanding", "acknowledge"],
                "constructive"
        ),

        # Case Identifying non_understanding
        (
                ["User: What are features?",
                 "Agent:The model uses different features to make a prediction."],
                "Sorry what?",
                ["signal_non_understanding"],
                "active"
        ),
        # Case signal_understanding, signal_partial_understanding
        (
                ["User: Why is income important?",
                 "Agent: Income directly influences the prediction outcome."],
                "I understand income is important, but how exactly does it influence the outcome?",
                ["signal_understanding", "signal_partial_understanding"],
                "constructive"
        ),

        # Case signal_non_understanding
        (
                ["User: Why is the prediction under 50k?",
                 "Agent: Because the combination of age and marital status strongly favors this class."],
                "I'm confused, how can age and marital status both favor under 50k?",
                ["signal_non_understanding"],
                "constructive"
        ),

        # Case 3: signal_understanding
        (
                ["User: How does marital status impact the prediction?",
                 "Agent: Marital status correlates with income levels, affecting the prediction."],
                "Are you sure marital status is a reliable indicator for income?",
                ["signal_understanding", "signal_partial_understanding"],
                "interactive"
        ),

        # Case 4: signal_understanding
        (
                ["User: What features contribute most to the prediction?",
                 "Agent: Age and income are the top contributors."],
                "If income is a top feature, how does it interact with age in the prediction?",
                ["signal_understanding"],
                "constructive"
        ),

        # Additional Cases for Complexity and Diversity

        # Case 7: signal_understanding
        (
                ["User: What are the most important features for this prediction?",
                 "Agent: Age and income are the top features, with age having slightly more influence."],
                "Can you explain why age is more important than income? How would changing age affect the prediction?",
                ["signal_understanding"],
                "interactive"
        ),

        # Case 8: signal_non_understanding
        (
                ["User: Why does marital status influence predictions?",
                 "Agent: Marital status is correlated with financial responsibility, which affects income prediction."],
                "I don't see how marital status strongly correlates with financial responsibility. Isn't this a flawed assumption?",
                ["signal_non_understanding"],
                "interactive"
        ),

        # Case 9: signal_understanding
        (
                ["User: Does the model handle outliers in the data?",
                 "Agent: Outliers are adjusted using a smoothing function to reduce their impact."],
                "What criteria does the model use to identify outliers? And how does smoothing affect the results?",
                ["signal_understanding"],
                "constructive"
        )
    ]
)
def test_monitor_prompt(chat_history, user_message, expected_displays, expected_engagement):
    monitor_prompt = PromptTemplate(get_monitor_prompt_template().format(
        chat_history=chat_history,
        user_message=user_message,
        understanding_displays=understanding_displays.as_text(),
        modes_of_engagement=engagement_modes.get_modes_as_text()
    ))

    monitor_result = llm.structured_predict(MonitorResultModel, monitor_prompt)

    logger.info(f"Monitor result: {monitor_result}.\n")
    # Assert the types
    assert isinstance(monitor_result.reasoning, str), "Reasoning should be a string."
    assert isinstance(monitor_result.understanding_displays, list), "understanding_displays should be a list of labels."
    # Asstert at least one of the expected results
    assert any(item in monitor_result.understanding_displays for item in expected_displays), \
        f"Expected at least one of {expected_displays}, but got {monitor_result.understanding_displays}"
    assert monitor_result.mode_of_engagement == expected_engagement, \
        f"Expected mode of engagement to be {expected_engagement}, but got {monitor_result.mode_of_engagement}"


@pytest.mark.parametrize(
    "chat_history, user_message, monitor_result, last_shown_explanations, expected_changes",
    [

        # Case Combination of two explanations, one understood and one partially
        (["User: Why not over 50k? 40 hours work should be enough",
          "Agent: The model predicts 'under 50k' because certain features of the instance, like 'No Investment' and "
          "'Age 23', strongly support this prediction. Even though 40 hours of work is significant, other factors like "
          "'Middle School' education level and 'No Investment' outcome also play a role. If you're curious about how "
          "the prediction might change, for example, improving 'Work Life Balance' to 'Good' and having a 'Major Gain' "
          "in investments could lead to a different prediction. Would you like to explore more about how these features "
          "influence the prediction?",
          "User: Yes",
          "Agent: Let's explore why the model predicted 'under 50k' and what could support an 'over 50k' prediction. "
          "For this instance, being 'Married' is a strong factor that could support an 'over 50k' prediction, with a "
          "significant importance of 35%. Additionally, working in an 'Admin' occupation also supports 'over 50k', "
          "though to a lesser extent, with an importance of 2%. However, the current prediction is 'under 50k' because "
          "other factors, like 'Middle School' education and 'No Investment', weigh more heavily in the model's decision."
          " Would you like to know more about how these features interact or explore other aspects of the prediction?"],
         "I don't get it",
         ["signal_non_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfOver50k"}],
         [{'state': ExplanationState.NOT_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfOver50k'}]),

        # Case Combination of two explanations, one understood and one partially
        (["User: Why is it not over 50k?",
          "Agent: The influence of the attributes in favour of 50k is not big enough and the attributes Investment "
          "Outcome and Age are in favour of under 50k."
          "The prediction would be 'over 50k' if Investment Outcome was Major Gain (above 5k$)."],
         "I thought Marital Status is most important?",
         ["signal_partial_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfUnder50k"}, {"CeterisParibus": "PossibleClassFlips"}],
         [{'state': ExplanationState.PARTIALLY_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfUnder50k'},
          {'state': ExplanationState.UNDERSTOOD.value, 'explanation_name': 'CeterisParibus',
           'step': 'PossibleClassFlips'}]),

        # Case Partial Understanding
        (["User: What is the most important feature in favour of the current prediction?",
          "Agent: The most important feature in favour of predicting 'under 50k' is Investment Outcome with an "
          "importance of 20 percent."],
         "I thought Marital Status is most important?",
         ["signal_partial_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfUnder50k"}],
         [{'state': ExplanationState.PARTIALLY_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfUnder50k'}]),

        # Case Misunderstood - single feature misunderstanding
        (["User: Why is the person not over 50k?",
          "Agent: The prediction that the person is under 50k is influenced by several features. For instance, their "
          "age of 23 and education level of 'Middle School' are significant factors. Generally, younger individuals "
          "and those with lower education levels tend to earn less. Additionally, their occupation as 'Admin' and a "
          "work-life balance rated as 'Fair' may also contribute to lower income levels. Understanding how these "
          "features interact helps clarify why the model predicts an income under 50k."],
         "Why is work-life balance important?",
         ["signal_partial_understanding"],
         [{"FeatureImportances": "Concept"}],
         [{'state': ExplanationState.UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'Concept'},
          {'state': ExplanationState.PARTIALLY_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'Concept'}
          ]),

        # Case Misunderstood - single feature misunderstanding
        (["User: What is the most important feature?",
          "Agent: The most important feature is Marital Status with an importance of 35 percent. This attribute is in "
          "favour of the opposite prediction, which is over 50k."],
         "So the prediction is over 50k?",
         ["signal_non_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfOver50k"}],
         [{'state': ExplanationState.NOT_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfOver50k'}]),

        # Case Misunderstood - misunderstanding of a single feature
        (["User: What is the most important feature?",
          "Agent: The most important feature is Marital Status with an importance of 35 percent. "
          "This attribute is in favour of the opposite prediction, which is over 50k."],
         "If the most important feature is in favour of the other class, why is it still under 50k?",
         ["signal_partial_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfOver50k"}],
         [{'state': ExplanationState.PARTIALLY_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfOver50k'}]),

        # Case Understood after scaffolding
        (history_after_scaffolding[:-1],
         history_after_scaffolding[-1],
         ["signal_understanding"],
         [{"FeatureImportances": "Concept"}, {"FeatureImportances": "FeaturesInFavourOfUnder50k"},
          {"ScaffoldingStrategy": "ElicitingFeedback"}],
         [{'state': ExplanationState.UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfUnder50k'}]),

        # Case Misunderstood - misunderstanding of a single feature
        (["User: What is the most important feature?",
          "Agent: The most important feature is Marital Status with an importance of 35 percent. "
          "This attribute is in favour of the opposite prediction, which is over 50k."],
         "If the most important feature is in favour of the other class, why is it still under 50k?",
         ["signal_partial_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfOver50k"}],
         [{'state': ExplanationState.PARTIALLY_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfOver50k'}]),

        # Case 2: Understood - understanding a counterfactual explanation
        (["User: How can I change the prediction to the opposite class?",
          "Agent: To change the prediction to 'over 50k', you can change the 'Investment Outcome' to 'Major Gain (above 5k$)'."],
         "Okay, so investment outcome is important for the prediction.",
         ["signal_understanding"],
         [{"Counterfactuals": "ImpactSingleFeature"}],
         [{'state': ExplanationState.UNDERSTOOD.value, 'explanation_name': 'Counterfactuals',
           'step': 'ImpactSingleFeature'}]),

        # Case 5: In Explanation - seeking deeper understanding
        (["User: What is the most important feature in favour of the current prediction?",
          "Agent: The most important feature in favour of predicting 'under 50k' is Investment Outcome with an "
          "importance of 20 percent."],
         "Does any feature have a higher importance?",
         ["signal_partial_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfUnder50k"}],
         [{'state': ExplanationState.PARTIALLY_UNDERSTOOD.value, 'explanation_name': 'FeatureImportances',
           'step': 'FeaturesInFavourOfUnder50k'}]),
    ]
)
def test_analyze_prompt(chat_history, user_message, monitor_result, last_shown_explanations, expected_changes):
    for shown_exp in last_shown_explanations:
        if isinstance(shown_exp, str):
            shown_exp = json.loads(shown_exp)
        exp_name = list(shown_exp.keys())[0]
        exp_step = shown_exp[exp_name]
        user_model.update_explanation_step_state(exp_name, exp_step, ExplanationState.SHOWN)

    analyze_prompt = PromptTemplate(get_analyze_prompt_template().format(
        domain_description=domain_description,
        feature_names=feature_names,
        instance=instance,
        predicted_class_name=predicted_class_name,
        chat_history=chat_history,
        understanding_displays=understanding_displays.as_text(),
        user_model=user_model.get_state_summary(as_dict=False),
        last_shown_explanations=last_shown_explanations,
        user_message=user_message,
        monitor_display_result=monitor_result.__str__(),
        monitor_cognitive_state="constructive",
        explanation_plan=user_model.get_complete_explanation_collection(as_dict=False),
    ))

    response = llm.structured_predict(output_cls=AnalyzeResult, prompt=analyze_prompt)
    logger.info(f"Response: {response}")
    assert isinstance(response.reasoning, str)
    assert isinstance(response.model_changes, list)
    assert any(item in expected_changes for item in response.model_changes), \
        f"Expected at least one of {expected_changes}, but got {response.model_changes}"


@pytest.mark.parametrize(
    "chat_history, user_message, monitor_result, last_shown_explanations, expected_plan, expected_step",
    [
        # Case elicit user knowledge about feature importances
        ([],
         "Why not over 50k?",
         ["signal_non_understanding"],
         [],
         [{"explanation_name": "FeatureImportances", "step": "Concept"},
          {"explanation_name": "FeatureImportances", "step": "FeaturesInFavourOfUnder50k"}, ],
         [{"explanation_name": "FeatureImportances",
           "step_name": "Concept",
           "communication_steps": [
               "Elicit if user knows how ML uses features to make predictions and wait for confirmation.",
               "Explain that features have different weight."]}]),

        # Case create new explanation plan
        ([],
         "What are feature importances?",
         ["signal_non_understanding"],
         [],
         [{"explanation_name": "FeatureImportances", "step": "Concept"}],
         [{"explanation_name": "FeatureImportances",
           "step_name": "Concept",
           "communication_steps": [
               "Elicit if user knows how ML uses features to make predictions and wait for confirmation.",
               "Explain that features have different weight."]}]),

        # Case Simple misunderstanding of a single feature
        (["User: What are feature importances?",
          "Agent:The model uses different features to make a prediction and assigns importance to each feature."],
         "Sorry what?",
         ["signal_non_understanding"],
         [{"FeatureImportances": "Concept"}],
         [{"explanation_name": "ScaffoldingStrategy", "step": "Reformulating"}],
         []),

        # Case Concept Explanation
        (["User: Why not over 50k?",
          "Agent: The model predicts 'under 50k' because certain features strongly support this outcome. For instance, "
          "having 'No Investment' is a significant factor, contributing 26% to the prediction. Being 23 years old and "
          "having a 'Middle School' education also support this prediction. On the other hand, being 'Married' is a "
          "strong factor for predicting 'over 50k', but it wasn't enough to outweigh the other features. If you're "
          "curious about how the prediction might change, altering features like 'Investment Outcome' to 'Major Gain' "
          "could potentially flip the prediction to 'over 50k'. Would you like to explore more about how these features "
          "influence the prediction?"],
         "User: What are features?",
         ["signal_non_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfOver50k"},
          {"FeatureImportances": "FeaturesInFavourOfUnder50k"},
          {"Counterfactuals": "Concept"},
          {"Counterfactuals": "ImpactMultipleFeatures"}],
         [{"explanation_name": "Features", "step": "Definition"},
          {"explanation_name": "FeaturesDefinition", "step": "MachineLearningModel"},
          {"explanation_name": "FeaturesDefinition", "step": "Data"}],
         []),

        # Case Simple understanding of counterfactual
        (["User: How can I change the prediction to the opposite class?",
          "Agent: To change the prediction to 'over 50k', you can change the 'Investment Outcome' to 'Major Gain (above 5k$)'."],
         "Okay, so investment outcome is important for the prediction.",
         ["signal_understanding"],  # Added monitor_result
         [{"Counterfactuals": "ImpactSingleFeature"}],
         [{"explanation_name": "Counterfactuals", "step": "ImpactMultipleFeatures"}],
         []),

        # Case Simple misunderstanding of a single feature
        (["User: What is the most important feature?",
          "Agent: The most important feature is Marital Status with an importance of 35 percent. This attribute is in favour of the opposite prediction, which is over 50k."],
         "If the most important feature is in favour of the other class, why is it still under 50k?",
         ["signal_partial_understanding"],
         [{"FeatureImportances": "FeaturesInFavourOfOver50k"}],
         [{"explanation_name": "FeatureImportances", "step": "Concept"},
          {"explanation_name": "FeatureImportances", "step": "FeaturesInFavourOfUnder50k"},
          {"explanation_name": "ScaffoldingStrategy", "step": "Reformulating"}],
         []),
    ]
)
def test_plan_prompt(chat_history, user_message, monitor_result, last_shown_explanations, expected_plan,
                     expected_step):
    for shown_exp in last_shown_explanations:
        if isinstance(shown_exp, str):
            shown_exp = json.loads(shown_exp)
        exp_name = list(shown_exp.keys())[0]
        exp_step = shown_exp[exp_name]
        user_model.update_explanation_step_state(exp_name, exp_step, ExplanationState.SHOWN)

    # Plan the next steps
    reasoning_prompt = PromptTemplate(get_plan_prompt_template().format(
        domain_description=domain_description,
        understanding_display=monitor_result,
        cognitive_state="constructive",
        feature_names=feature_names,
        instance=instance,
        predicted_class_name=predicted_class_name,
        chat_history=chat_history,
        user_model=user_model.get_state_summary(as_dict=False),
        user_message=user_message,
        explanation_plan=user_model.get_complete_explanation_collection(as_dict=False),
        previous_plan=[],
        last_explanation=[]
    ))
    plan_result = llm.structured_predict(output_cls=PlanResultModel, prompt=reasoning_prompt)
    # If the plan has new explanations, add them to the user model
    if len(plan_result.explanation_plan) > 0:
        user_model.add_explanations_from_plan_result(plan_result.new_explanations)

    # Extract exp_name and step tuples from explanation_plan
    next_explanations = []
    for exp in plan_result.explanation_plan:
        next_explanations.append({"explanation_name": exp.explanation_name, "step": exp.step})

    logger.info(f"Plan result: {plan_result}.\n")
    # Assert the types
    # assert isinstance(plan_result.reasoning, str), "Reasoning should be a string."
    assert isinstance(next_explanations, list), "Next explanations should be a list of tuples."
    assert any(item in next_explanations for item in expected_plan), \
        f"Expected at least one of {expected_plan}, but got {next_explanations}"


@pytest.mark.parametrize(
    "chat_history, user_message, suggested_plan, suggested_communication_steps, monitor_result, shown_explanations",
    [
        # Case: Elicit user knowledge about feature importances
        ([],
         "Why not over 50k?",
         [("FeatureImportances", "Concept"), ("FeatureImportances", "FeaturesInFavourOfUnder50k")],
         ["Elicit if user knows how ML uses features to make predictions and wait for confirmation.",
          "Explain the concept of feature importance and its role in model predictions."],
         ["signal_non_understanding"],
         []),

    ]
)
def test_execute_prompt(chat_history, user_message, suggested_plan, suggested_communication_steps, monitor_result,
                        shown_explanations):
    for shown_exp in shown_explanations:
        if isinstance(shown_exp, str):
            shown_exp = json.loads(shown_exp)
        if isinstance(shown_exp, tuple):
            exp_name, exp_step = shown_exp
        else:
            exp_name = list(shown_exp.keys())[0]
            exp_step = shown_exp[exp_name]
        user_model.update_explanation_step_state(exp_name, exp_step, ExplanationState.SHOWN)

    # Turn the suggested plan into a list of ChosenExplanationModel objects
    suggested_plan = [ChosenExplanationModel(explanation_name=exp_name, step=exp_step) for exp_name, exp_step in
                      suggested_plan]

    suggested_plan_info = user_model.get_string_explanations_from_plan(suggested_plan)

    suggested_communication_step = suggested_communication_steps[0]

    execute_prompt = PromptTemplate(get_execute_prompt_template().format(
        domain_description=domain_description,
        feature_names=feature_names,
        instance=instance,
        predicted_class_name=predicted_class_name,
        chat_history=chat_history,
        user_model=user_model.get_state_summary(as_dict=False),
        user_message=user_message,
        plan_result=suggested_plan_info,
        next_exp_content=suggested_communication_step,
        monitor_display_result=monitor_result,
        monitor_cognitive_state="active",
    ))

    execute_result = llm.structured_predict(ExecuteResult, execute_prompt)
    logger.info(f"Execute result: {execute_result}.")
    assert isinstance(execute_result.reasoning, str)
