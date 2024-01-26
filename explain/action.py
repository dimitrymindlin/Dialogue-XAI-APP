"""Executes actions for parsed canonical utterances.

This file implements routines to take actions in the conversation, returning
outputs to the user. Actions in the conversation are called `operations` and
are things like running an explanation or performing filtering.
"""
from flask import Flask
from jinja2 import Environment, FileSystemLoader
import numpy as np

from explain.actions.explanation import explain_local_feature_importances, explain_cfe, \
    get_feature_importance_by_feature_id, explain_cfe_by_given_features, \
    explain_anchor_changeable_attributes_without_effect, explain_feature_statistic, explain_feature_importances_as_plot, \
    explain_global_feature_importances, explain_ceteris_paribus
from explain.actions.filter import filter_operation
from explain.actions.prediction_likelihood import predict_likelihood
from explain.conversation import Conversation
from explain.actions.get_action_functions import get_all_action_functions_map

app = Flask(__name__)


def run_action(conversation: Conversation,
               parse_tree,
               parsed_string: str,
               actions=get_all_action_functions_map(),
               build_temp_dataset: bool = True) -> str:
    """Runs the action and updates the conversation object

    Arguments:
        build_temp_dataset: Whether to use the temporary dataset stored in the conversation
                            or to rebuild the temporary dataset from scratch.
        actions: The set of avaliable actions
        parsed_string: The grammatical text
        conversation: The conversation object, see `conversation.py`
        parse_tree: The parse tree of the canonical utterance. Note, currently, this is not used,
                    and we compute the actions from the parsed text.
    """
    if parse_tree:
        pretty_parse_tree = parse_tree.pretty()
        app.logger.info(f'Parse tree {pretty_parse_tree}')

    return_statement = ''

    # Will rebuilt the temporary dataset if requested (i.e, for filtering from scratch)
    if build_temp_dataset:
        conversation.build_temp_dataset()

    parsed_text = parsed_string.split(' ')
    is_or = False

    for i, p_text in enumerate(parsed_text):
        if parsed_text[i] in actions:
            action_return, action_status = actions[p_text](
                conversation, parsed_text, i, is_or=is_or)
            return_statement += action_return

            # If operation fails, return error output to user
            if action_status == 0:
                break

            # This is a bit ugly but basically if an or occurs
            # we hold onto the or until we need to flip it off, i.e. filtering
            # happens
            if is_or is True and actions[p_text] == 'filter':
                is_or = False

        if p_text == 'or':
            is_or = True

    # Store 1-turn parsing
    conversation.store_last_parse(parsed_string)

    while return_statement.endswith("<br>"):
        return_statement = return_statement[:-len("<br>")]

    return return_statement


def run_action_by_id(conversation: Conversation,
                     question_id: int,
                     instance_id: int,
                     feature_id: int = None,
                     build_temp_dataset: bool = True,
                     instance_type_naming: str = "instance") -> str:
    """
    Runs the action selected by an ID instead of text parsing and updates the conversation object.

    conversation: Conversation, Conversation Object
    question_id: int, id of the question as defined in question_bank.csv
    instance_id: int, id of the instance that should be explained. Needed for local explanations
    feature_id: int, id of the feature name the question is about (if specified)
    build_temp_dataset: bool = True If building tmp_dataset is needed.
    """
    if build_temp_dataset:
        conversation.build_temp_dataset()

    # Create parse text as filter works with it
    parse_text = f"filter id {instance_id}".split(" ")
    _ = filter_operation(conversation, parse_text, 0)
    # Get tmp dataset to perform explanation on (here, single ID will be in tmp_dataset)
    data = conversation.temp_dataset.contents['X']
    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    feature_name = data.columns[feature_id]
    parse_op = f"ID {instance_id}"
    current_prediction = conversation.get_class_name_from_label(conversation.temp_dataset.contents['y'][instance_id])
    opposite_class = conversation.get_class_name_from_label(1 - conversation.temp_dataset.contents['y'][instance_id])

    # get_explanation_report(conversation, instance_id)

    if question_id == 0:
        # Which attributes does the model use to make predictions?
        return f"The model uses the following attributes to make predictions: {', '.join(list(data.columns))}."
    if question_id == 1:
        # Does the model include [feature X] when making the prediction?
        explanation = get_feature_importance_by_feature_id(conversation, data, regen, feature_id)
        answer = "Yes it does. "
        return answer + explanation[0]
    if question_id == 2:
        # How important is each attribute to the model's predictions?
        # Create full feature explanations
        # explanation = explain_feature_importances(conversation, data, parse_op, regen)
        explanation = explain_feature_importances_as_plot(conversation, data, parse_op, regen)
        return explanation[0]
    if question_id == 3:
        # How strong does [feature X] affect the prediction?
        explanation = get_feature_importance_by_feature_id(conversation, data, regen, feature_id)
        return explanation[0]
    if question_id == 4:
        # What are the top 3 important attributes for this prediction?
        parse_op = "top 3"
        explanation = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=True)
        answer = "Here are the 3 most important attributes for this prediction: <br><br>"
        return answer + explanation[0]
    """if question_id == 5:
        # Why did the model give this particular prediction for this person?
        explanation = explain_feature_importances(conversation, data, parse_op, regen,
                                                  return_full_summary=False)
        answer = "The prediction can be explained by looking at the most important attributes. <br>"
        answer += "Here are the top most important ones: <br>"
        return answer + explanation[0]"""
    if question_id == 5:
        # What attributes of this instance led the model to make this prediction?
        explanation = explain_local_feature_importances(conversation, data, parse_op, regen)
        answer = "The following attributes were most important for the prediction. "
        return answer + explanation[0]
    """if question_id == 6:
        # What would happen to the prediction if we changed [feature] for this instance?
        explanation = explain_cfe_by_given_features(conversation, data, [feature_name])
        if explanation[1] == 0:
            answer = explanation[0] + feature_name + "."
        else:
            answer = explanation[0]
        return answer"""
    if question_id == 7:
        # How should this instance change to get a different prediction?
        explanation = explain_cfe(conversation, data, parse_op, regen)
        explanation = "Here are some changes to the instance that would lead to a different prediction: <br>" + \
                      explanation[0]
        return explanation
    if question_id == 8:
        # How should this attribute change to get a different prediction?
        top_features_dict, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=False)
        explanation = explain_cfe_by_given_features(conversation, data, [feature_name], top_features_dict)
        return explanation
    if question_id == 9:
        # Which changes to this instance would still get the same prediction?
        explanation = explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen)
        return explanation[0]
    if question_id == 10:
        # Which maximum changes would not influence the class prediction?
        pass
    if question_id == 11:
        # What attributes must be present or absent to guarantee this prediction?
        explanation = explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen)
        return explanation[0]
    if question_id == 12:
        # How does the prediction change when this attribute changes? Ceteris Paribus
        # explanation = explain_ceteris_paribus(conversation, data, parse_op, regen)
        return f"This is a mocked answer to your question with id {question_id}."
    if question_id == 13:
        # 13;How common is the current values for this attribute?
        explanation = explain_feature_statistic(conversation, feature_name)
        return explanation
    if question_id == 20:
        # 20;Why is this instance predicted as [current prediction]?
        explanation = explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen)
        result_text = f"The {instance_type_naming} is predicted as <it>{current_prediction}</it>, because the "
        result_text = result_text + explanation[0]
        return result_text
    if question_id == 22:
        # 22;How is the model using the attributes in general to give an answer?
        explanation = explain_global_feature_importances(conversation)
        explanation_intro = "The model uses the attributes to inform its predictions by assigning each a level of importance. " \
                            f"The chart with bars indicates the general trend of importances across all f{instance_type_naming}s: longer bars correspond to attributes with a greater general influence on the model's decisions.<br>"
        return explanation_intro + explanation[0]
    if question_id == 23:
        # 23;Which are the most important attributes for the outcome of the instance?
        parse_op = "top 3 only_positive"
        explanation = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=True)
        answer = f"Here are the 3 <b>most</b> important factors for this {instance_type_naming}: <br><br>"
        return answer + explanation[0]

    if question_id == 24:
        # 24;What are the attributes and their impact for the current prediction of [curent prediction]?
        explanation = explain_feature_importances_as_plot(conversation, data, parse_op, regen, current_prediction)
        return explanation[0]
    if question_id == 25:
        # 25;What if I changed the value of a feature?; What if I changed the value of [feature selection]?;Ceteris Paribus
        explanation = explain_ceteris_paribus(conversation, data, feature_name)
        #intro = "The following graph shows the prediction score on the Y Axis when changing only the selected attribute. <br>"
        return  explanation[0] + f" <i>{opposite_class}</i>."
    if question_id == 27:
        # 27;What features are used the least for prediction of the current instance?; What attributes are used the least for prediction of the instance?
        parse_op = "least 3"
        explanation = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=True)
        return explanation[0]
    else:
        return f"This is a mocked answer to your question with id {question_id}."
    """if question_id == 12:
        # How does the prediction change when this attribute changes? Ceteris Paribus
        explanation = explain_ceteris_paribus(conversation, data, parse_op, regen)"""


def get_explanation_report(conversation,
                           instance_id: int):
    """
    Runs explanation methods on the current conversation and returns a static report.
    """

    parse_text = f"filter id {instance_id}".split(" ")
    _ = filter_operation(conversation, parse_text, 0)
    data = conversation.temp_dataset.contents['X']
    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    parse_op = f"ID {instance_id}"

    model_prediction_probas, _ = predict_likelihood(conversation, as_text=False)
    model_prediction = conversation.get_class_name_from_label(np.argmax(model_prediction_probas))
    opposite_class = conversation.get_class_name_from_label(np.argmin(model_prediction))
    feature_importances, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=False)
    # Turn list of values into int
    feature_importances = {key: round(float(value[0]), ndigits=3) for key, value in feature_importances.items()}
    counterfactual_strings, _ = explain_cfe(conversation, data, parse_op, regen)
    anchors_string, _ = explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen)

    # Fill static report template
    # Load md file
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)
    template = env.get_template('templates/static_report_template.md')

    markdown = template.render(
        model_prediction=model_prediction,
        instance_type="Applicant",
        feature_importance=feature_importances,
        opposite_class=opposite_class,
        counterfactuals=counterfactual_strings,
        anchors=anchors_string,
    )

    # Save the rendered Markdown to a file
    output_file = f'static_report_{instance_id}.md'
    with open(output_file, 'w') as file:
        file.write(markdown)

    new_point = data.copy()
    new_point.at[instance_id, 'Checking account'] = 3

    print()
