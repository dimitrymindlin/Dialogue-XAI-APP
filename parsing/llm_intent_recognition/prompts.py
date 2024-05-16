from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def get_template_with_full_descriptions():
    return f"""The user was presented an instance with different features.
        The machine learning model predicted a class. Given the user question about the model prediction, 
        decide which method fits best to answer it. There are standalone methods that work without requiring a feature 
        specification: {general_questions} and some that are specific to a feature: {feature_specific_q}.""" + """
        
    Here are definitions for the available methods:
    - whyExplanation: Explains which explanations are possible when the user asks a general why question.
    Best for questions like: "Why this prediction"?
    Answer example: "To understand the prediction, I can tell you about the most important attributes, or 
    which changes would have led to a different prediction."
    - greeting: Greets the user when he does not ask a specific question. If the user just says hi or how are you,
    tell him what questions you can answer.
    Best for questions like: "Hey, how are you?"
    Answer example: "Hello, I am an assistant to help you understand the prediction of the Machine Learning model.
    You can ask about the most or least important attributes, how certain changes in attributes influence the prediction
    or what alternative attributes would lead to a different prediction."
    - notXaiMethod: Used to answer questions that are not possible to answer with the previous methods. Could
    be that the user asks for a clarification on an explanation or questions related to the domain.
    Best for questions like: "What does it mean?"
    Answer example: "[Description of the questioned thing]".
    - followUp: This method is used when the user mentions a feature feature without specifying if
    he is interested in the feature change or feature statistics.
    Best for questions like: "And what about age?", "And income?", or "Okay, and what about education level?"
    Answer example: "Here is the same explanation method for the new feature."
    - anchor: Anchor explanations identify the minimal set of conditions and feature values that, when held constant, 
    ensure a specific prediction result does not change, irrespective of alterations to other features. 
    Best for questions like "What factors guarantee this prediction remains the same?"
    Answer example: "If 'age' and 'occupation' remain constant, the model's prediction of the income being above 
    50k will remain unchanged. These features are the anchors in this scenario."
    - shapAllFeatures: uses SHAP values to measure and visualize all feature's individual contribution to the 
    overall prediction, considering the marginal effect of each feature across all possible combinations. 
    This comprehensive approach provides a complete overview by showing the impact of all features of the instance.
    Best for questions like: "What is the strength of each feature?"
    Answer example: "Here is a visualization, showing the contribution of each feature..."
    - top3Features: This method focuses on identifying and visualizing the contributions of the top three 
    features that have the highest impact on the model's prediction, according to SHAP values. It simplifies the 
    explanation by concentrating on the most influential variables.
    Best for questions like: "Which features had the greatest impact on this prediction?"
    Answer example: "The top influences on the prediction were 'education level', 'occupation', and 'hours per week'."
    - least3Features: This method concentrates on the three features with the least impact on the model’s
    prediction, according to SHAP values. It provides insights into the features that have minimal influence on
    the outcome, which can be critical for understanding the robustness of the model or for identifying potential
    areas of model simplification.
    Best for questions like: "Which features had the least impact on this prediction?"
    Answer example: "The least influential features were 'marital status', 'race', and 'sex', each contributing minimally to the overall prediction according to their SHAP values."
    - counterfactualAnyChange: Provide possible feature alterations 
    to understand scenarios under which the model's prediction is changed. This method is suited for 
    exploring changes in the features of the current instance that would lead to a different prediction, 
    thereby clarifying why the current instance didn't classify as another category.
    Best for questions like: "Why is it not class [other class]?", or "In which case would be other class?"
    Answer example: "The prediction would switch from class A to class B if 'hours per week' increased by more than 10 hours,
     or 'age' was above 45 or ..."
    - featureStatistics: Provides a statistical summary or visual representation of the features in a dataset.
    It calculates the mean and standard deviation for numerical features, offering a clear quantitative 
    overview of data spread and central tendency.
    Best for questions like: "What are the typical values and distributions of 'age' in my dataset?"
    Answer example: "The mean age is 40 years with a standard deviation of 12.5.
    - ceterisParibus: This method examines the impact of changing one specific feature. The user explicitely
    asks for the change of a feature. It is designed to explore hypothetical scenarios, analyzing what would occur if a 
    single feature's value were different (e.g., higher or lower). 
    Best for questions like: "What would happen if marital status was different?" or "What if hours per week increased?"
     Answer example: "Changing 'marital status' from 'single' to 'married' would result in the income prediction 
     changing from below 50k to above 50k."

Decide which methods fits best and if its a feature specific one, also reply with the feature.
Immediately answer with the method and feature if applicable, without justification.
\n{format_instructions}\n{input}
"""


def get_xai_template_with_descriptions():
    return f"""The user was presented an instance with different features.
        The machine learning model predicted a class. Given the user question about the model prediction, 
        decide which method fits best to answer it. There are standalone methods that work without requiring a feature 
        specification: {general_questions} and some that are specific to a feature: {feature_specific_q}.""" + """
    
    REMEMBER: "method" MUST be one of the candidate method names specified below OR it can be null if the input is not well suited for any of the candidate prompts.
    REMEMBER: "feature_name" can be null if the question is not related to a specific feature.
    
    << CANDIDATE METHODS >>
    - anchor: Anchor explanations identify the minimal set of conditions and feature values that, when held constant, 
    ensure a specific prediction result does not change, irrespective of alterations to other features. 
    Best for questions like "What factors guarantee this prediction remains the same?"
    Answer example: "If feature1 and feature2 remain constant, the model's prediction will remain unchanged."
    - shapAllFeatures: uses SHAP values to measure and visualize all feature's individual contribution to the 
    overall prediction, considering the marginal effect of each feature across all possible combinations. 
    This comprehensive approach provides a complete overview by showing the impact of all features of the instance.
    Best for questions like: "What is the strength of each feature?"
    Answer example: "Here is a visualization, showing the contribution of each feature..."
    - top3Features: This method focuses on identifying and visualizing the contributions of the top three 
    features that have the highest impact on the model's prediction, according to SHAP values. It simplifies the 
    explanation by concentrating on the most influential variables.
    Best for questions like: "Which features had the greatest impact on this prediction?"
    Answer example: 'The top influences on the prediction were feature1, feature2, and feature3.'"
    - least3Features: This method concentrates on the three features with the least impact on the model’s
    prediction, according to SHAP values. It provides insights into the features that have minimal influence on
    the outcome, which can be critical for understanding the robustness of the model or for identifying potential
    areas of model simplification.
    Best for questions like: "Which features had the least impact on this prediction?"
    Answer example: 'The least influential features were feature1, feature2, and feature3, each contributing minimally 
    to the overall prediction according to their SHAP values.'
    - ceterisParibus: This method focuses on looking for the impact of changes in a feature requested by the 
    user. It is useful to investigate what would happen if some feature was different. i.e higher or lower.
     Best for questions like: "What would happen if feature1 was different?"
     Answer example: "Changing feature1 from value1 to value2 would result in the prediction changing from class1 to class2."
    - counterfactualAnyChange: Provide possible feature alterations to understand scenarios under which the model's 
    prediction is changed. This method is suited for exploring changes in the features of the current instance that 
    would lead to a different prediction, thereby clarifying why the current instance didn't classify as another category. 
    Best for questions like: 'Why is it not class2?', or 'In which case would be other class?'
    Answer example: 'The prediction would switch from class1 to class2 if feature1 increased by value1, or feature2 was 
    above value2 or ...'
    - featureStatistics: Provides a statistical summary or visual representation of the features in a dataset.
    It calculates the mean and standard deviation for numerical features, offering a clear quantitative 
    overview of data spread and central tendency, giving intuition about whether a certain current value is 
    particularly high, low or average. 
    Best for questions like: "What are the typical values and distributions of feature1 in my dataset?"
    Answer example: "The mean of feature1 is value1 with a standard deviation of value2."

\n{format_instructions}

REMEMBER: Do not justify the choice of the method or feature, just provide the method and feature.

<< User Question >>
\n{input}

<< Answer >>
"""


general_questions = [
    "anchor",
    "shapAllFeatures",
    "top3Features",
    "least3Features",
    "counterfactualAnyChange"]
feature_specific_q = [
    "ceterisParibus",
    "featureStatistics",
]
extra_dialogue_intents = [
    "followUp",
    "whyExplanation",
    "notXaiMethod",
    "greeting"
]
possible_categories = general_questions + feature_specific_q + extra_dialogue_intents
possible_features = [
    'Age',
    'EducationLevel',
    'MaritalStatus',
    'Occupation',
    'WeeklyWorkingHours',
    'WorkLifeBalance',
    'InvestmentOutcome']
question_to_id_mapping = {
    "top3Features": 23,
    "anchor": 11,
    "shapAllFeatures": 24,
    "least3Features": 27,
    "ceterisParibus": 25,
    "featureStatistics": 13,
    "counterfactualAnyChange": 7,
    "followUp": 0,
    "whyExplanation": 1,
    "notXaiMethod": 100,
    "greeting": 99,
    "null": -1,
}
response_schemas = [
    ResponseSchema(name="method_name", description="name of the method to answer the user question."),
    ResponseSchema(name="feature", description="Feature that the user mentioned, can be null."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

ROUTING_TASK_PROMPT = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. Copy the input into the next_inputs field.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""
