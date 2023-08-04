"""The main script that controls conversation logic.

This file contains the core logic for facilitating conversations. It orchestrates the necessary
routines for setting up conversations, controlling the state of the conversation, and running
the functions to get the responses to user inputs.
"""
import json
import pickle
from random import seed as py_random_seed
import secrets
from typing import List
import re

from jinja2 import Environment, FileSystemLoader
import numpy as np
import pandas as pd
import torch

from flask import Flask
import gin

from create_experiment_data.experiment_helper import ExperimentHelper
from explain.action import run_action, run_action_by_id
from explain.actions.explanation import explain_cfe, explain_cfe_by_given_features
from explain.conversation import Conversation
from explain.decoder import Decoder
from explain.explanation import MegaExplainer
from explain.explanations.anchor_explainer import TabularAnchor
from explain.explanations.ceteris_paribus import CeterisParibus
from explain.explanations.dice_explainer import TabularDice
from explain.explanations.diverse_instances import DiverseInstances
from explain.explanations.feature_statistics_explainer import FeatureStatisticsExplainer
from explain.parser import Parser, get_parse_tree
from explain.prompts import Prompts
from explain.utils import read_and_format_data
from explain.write_to_log import log_dialogue_input

app = Flask(__name__)


@gin.configurable
def load_sklearn_model(filepath):
    """Loads a sklearn model."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


@gin.configurable
class ExplainBot:
    """The ExplainBot Class."""

    def __init__(self,
                 model_file_path: str,
                 dataset_file_path: str,
                 background_dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: List[str],
                 numerical_features: List[str],
                 remove_underscores: bool,
                 name: str,
                 parsing_model_name: str = "nearest-neighbor",
                 seed: int = 0,
                 prompt_metric: str = "cosine",
                 prompt_ordering: str = "ascending",
                 t5_config: str = None,
                 use_guided_decoding: bool = True,
                 feature_definitions: dict = None,
                 skip_prompts: bool = False,
                 categorical_mapping_path: str = None,
                 feature_tooltip_mapping=None,
                 actionable_features=None,
                 feature_units=None):
        """The init routine.

        Arguments:
            model_file_path: The filepath of the **user provided** model to explain. This model
                             should end with .pkl and support sklearn style functions like
                             .predict(...) and .predict_proba(...)
            dataset_file_path: The path to the dataset used in the conversation. Users will understand
                               the model's predictions on this dataset.
            background_dataset_file_path: The path to the dataset used for the 'background' data
                                          in the explanations.
            dataset_index_column: The index column in the data. This is used when calling
                                  pd.read_csv(..., index_col=dataset_index_column)
            target_variable_name: The name of the column in the dataset corresponding to the target,
                                  i.e., 'y'
            categorical_features: The names of the categorical features in the data. If None, they
                                  will be guessed.
            numerical_features: The names of the numeric features in the data. If None, they will
                                be guessed.
            remove_underscores: Whether to remove underscores in the feature names. This might help
                                performance a bit.
            name: The dataset name
            parsing_model_name: The name of the parsing model. See decoder.py for more details about
                                the allowed models.
            seed: The seed
            prompt_metric: The metric used to compute the nearest neighbor prompts. The supported options
                           are cosine, euclidean, and random
            prompt_ordering:
            t5_config: The path to the configuration file for t5 models, if using one of these.
            skip_prompts: Whether to skip prompt generation. This is mostly useful for running fine-tuned
                          models where generating prompts is not necessary.
            categorical_mapping_path: Path to json mapping for each col that assigns a categorical var to an int.
            feature_tooltip_mapping: A mapping from feature names to tooltips. This is used to display tooltips
                                        in the UI.
            actionable_features: A list of features that can be changed (actionable features)
            feature_units: A mapping from feature names to units. This is used to display units in the UI.
        """

        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)
        torch.manual_seed(seed)

        self.bot_name = name

        # Prompt settings
        self.prompt_metric = prompt_metric
        self.prompt_ordering = prompt_ordering
        self.use_guided_decoding = use_guided_decoding
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.feature_tooltip_mapping = feature_tooltip_mapping
        self.actionable_features = actionable_features
        self.feature_units = feature_units

        # A variable used to help file uploads
        self.manual_var_filename = None

        self.decoding_model_name = parsing_model_name

        # Initialize completion + parsing modules
        """
        app.logger.info(f"Loading parsing model {parsing_model_name}...")
        self.decoder = Decoder(parsing_model_name,
                               t5_config,
                               use_guided_decoding=self.use_guided_decoding,
                               dataset_name=name)"""
        self.decoder = None

        self.data_instances = []
        self.current_instance = []

        # Initialize parser + prompts as None
        # These are done when the dataset is loaded
        self.prompts = None
        self.parser = None

        # Set up the conversation object
        self.conversation = Conversation(eval_file_path=dataset_file_path,
                                         feature_definitions=feature_definitions)

        # Load the model into the conversation
        self.load_model(model_file_path)

        # Load categorical mapping
        if categorical_mapping_path is not None:
            with open(categorical_mapping_path, "r") as f:
                categorical_mapping = json.load(f)
                self.categorical_mapping = {int(k): v for k, v in categorical_mapping.items()}
        else:
            self.categorical_mapping = None

        # Load the dataset into the conversation
        self.load_dataset(dataset_file_path,
                          dataset_index_column,
                          target_variable_name,
                          categorical_features,
                          numerical_features,
                          remove_underscores,
                          store_to_conversation=True,
                          skip_prompts=skip_prompts)

        background_dataset, background_y_values = self.load_dataset(background_dataset_file_path,
                                                                    dataset_index_column,
                                                                    target_variable_name,
                                                                    categorical_features,
                                                                    numerical_features,
                                                                    remove_underscores,
                                                                    store_to_conversation=False)

        # Load the explanations
        self.load_explanations(background_dataset=background_dataset,
                               y_values=background_y_values,
                               categorical_mapping=self.categorical_mapping)

    def get_next_instance(self):
        """
        Returns the next instance in the data_instances list if possible.
        """
        if len(self.data_instances) == 0:
            self.load_data_instances()  # TODO: Infinity loop - Where is experiment end determined?
        self.current_instance = self.data_instances.pop(0)
        return self.current_instance

    def get_feature_tooltips(self):
        """
        Returns the feature tooltips for the current dataset.
        """
        return self.feature_tooltip_mapping

    def init_loaded_var(self, name: bytes):
        """Inits a var from manual load."""
        self.manual_var_filename = name.decode("utf-8")

    def get_questions_and_attributes(self):
        """
        Returns the questions and attributes for the current dataset.
        """
        question_pd = pd.read_csv(self.conversation.question_bank_path, delimiter=";")
        feature_names = list(self.conversation.get_var("dataset").contents['X'].columns)
        answer_dict = {
            "general_questions": [{'id': row['q_id'], 'question': row['paraphrased']} for _, row in
                                  question_pd[question_pd["question_type"] == "general"].iterrows()],
            # question_pd[question_pd["question_type"] == "general"][["q_id", "paraphrased"]],
            "feature_questions": [{'id': row['q_id'], 'question': row['paraphrased']} for _, row in
                                  question_pd[question_pd["question_type"] == "feature"].iterrows()],
            # list(question_pd[question_pd["question_type"] == "feature"][["q_id", "paraphrased"]].values),
            "feature_names": [{'id': feature_id, 'feature_name': feature_name} for feature_id, feature_name in
                              enumerate(feature_names)],
        }
        return answer_dict

    def load_explanations(self, background_dataset,
                          y_values=None,
                          categorical_mapping=None):
        """Loads the explanations.

        If set in gin, this routine will cache the explanations.

        Arguments:
            background_dataset: The background dataset to compute the explanations with.
        """
        app.logger.info("Loading explanations into conversation...")

        # This may need to change as we add different types of models
        pred_f = self.conversation.get_var('model_prob_predict').contents
        model = self.conversation.get_var('model').contents
        data = self.conversation.get_var('dataset').contents['X']
        categorical_f = self.conversation.get_var('dataset').contents['cat']
        numeric_f = self.conversation.get_var('dataset').contents['numeric']

        # Load lime tabular explanations
        mega_explainer = MegaExplainer(prediction_fn=pred_f,
                                       data=background_dataset,
                                       cat_features=categorical_f,
                                       class_names=self.conversation.class_names,
                                       categorical_mapping=self.categorical_mapping)
        mega_explainer.get_explanations(ids=list(data.index),
                                        data=data)
        message = (f"...loaded {len(mega_explainer.cache)} mega explainer "
                   "explanations from cache!")
        app.logger.info(message)
        # Load lime dice explanations
        tabular_dice = TabularDice(model=model,
                                   data=data,
                                   num_features=numeric_f,
                                   class_names=self.conversation.class_names,
                                   categorical_mapping=self.categorical_mapping)
        tabular_dice.get_explanations(ids=list(data.index),
                                      data=data)
        message = (f"...loaded {len(tabular_dice.cache)} dice tabular "
                   "explanations from cache!")
        app.logger.info(message)

        # Load diverse instances (explanations)
        diverse_instances_explainer = DiverseInstances(
            lime_explainer=mega_explainer.mega_explainer.explanation_methods['lime_0.75'])
        diverse_instance_ids = diverse_instances_explainer.get_instance_ids_to_show(data=data)
        message = (f"...loaded {len(diverse_instance_ids)} diverse instance ids "
                   "from cache!")

        # Make new list of dicts {id: instance_dict} where instance_dict is a dict with column names as key and values as values.
        diverse_instances = [{"id": i, "values": data.loc[i].to_dict()} for i in diverse_instance_ids]

        app.logger.info(message)

        # Load anchor explanations
        # categorical_names = create_feature_values_mapping_from_df(data, categorical_f)
        tabular_anchor = TabularAnchor(model=model,
                                       data=data,
                                       categorical_names=self.categorical_mapping,
                                       class_names=self.conversation.class_names,
                                       feature_names=list(data.columns))
        tabular_anchor.get_explanations(ids=list(data.index),
                                        data=data)

        # Load feature statistics explanations
        feature_statistics_explainer = FeatureStatisticsExplainer(data,
                                                                  self.numerical_features,
                                                                  feature_names=list(data.columns),
                                                                  rounding_precision=self.conversation.rounding_precision,
                                                                  categorical_mapping=self.categorical_mapping)

        # Load Ceteris Paribus Explanations
        """ceteris_paribus_explainer = CeterisParibus(model=model,
                                                   background_data=background_dataset,
                                                   ys=y_values,
                                                   class_names=self.conversation.class_names,
                                                   feature_names=list(data.columns))
        ceteris_paribus_explainer.get_explanations(ids=list(data.index),
                                                   data=data)"""

        # Add all the explanations to the conversation
        self.conversation.add_var('mega_explainer', mega_explainer, 'explanation')
        self.conversation.add_var('tabular_dice', tabular_dice, 'explanation')
        self.conversation.add_var('tabular_anchor', tabular_anchor, 'explanation')
        # list of dicts {id: instance_dict} where instance_dict is a dict with column names as key and values as values.
        self.conversation.add_var('diverse_instances', diverse_instances, 'diverse_instances')
        self.conversation.add_var('feature_statistics_explainer', feature_statistics_explainer, 'explanation')

        # Load Experiment Helper
        helper = ExperimentHelper(self.conversation, self.categorical_mapping, self.categorical_features)
        self.conversation.add_var('experiment_helper', helper, 'experiment_helper')

    def load_data_instances(self):
        # dataset_pd = self.conversation.get_var("dataset").contents['X']
        diverse_instances = self.conversation.get_var("diverse_instances").contents
        instance_results = []
        for instance in diverse_instances:
            id = instance['id']
            # current_instance = list(dataset_pd.loc[id].values)
            instance_result_dict = {}
            for i, (feature_name, val) in enumerate(instance['values'].items()):
                if i in self.categorical_mapping:
                    instance_result_dict[feature_name] = self.categorical_mapping[i][val]
                else:
                    instance_result_dict[feature_name] = val
            instance_results.append((id, instance_result_dict))
        self.data_instances = instance_results

    def load_model(self, filepath: str):
        """Loads a model.

        This routine loads a model into the conversation
        from a specified file path. The model will be saved as a variable
        names 'model' in the conversation, overwriting an existing model.

        The routine determines the type of model from the file extension.
        Scikit learn models should be saved as .pkl's and torch as .pt.

        Arguments:
            filepath: the filepath of the model.
        Returns:
            success: whether the model was saved successfully.
        """
        app.logger.info(f"Loading inference model at path {filepath}...")
        if filepath.endswith('.pkl'):
            model = load_sklearn_model(filepath)
            self.conversation.add_var('model', model, 'model')
            self.conversation.add_var('model_prob_predict',
                                      model.predict_proba,
                                      'prediction_function')
        else:
            # No other types of models implemented yet
            message = (f"Models with file extension {filepath} are not supported."
                       " You must provide a model stored in a .pkl that can be loaded"
                       f" and called like an sklearn model.")
            raise NameError(message)
        app.logger.info("...done")
        return 'success'

    def load_dataset(self,
                     filepath: str,
                     index_col: int,
                     target_var_name: str,
                     cat_features: list[str],
                     num_features: list[str],
                     remove_underscores: bool,
                     store_to_conversation: bool,
                     skip_prompts: bool = False):
        """Loads a dataset, creating parser and prompts.

        This routine loads a dataset. From this dataset, the parser
        is created, using the feature names, feature values to create
        the grammar used by the parser. It also generates prompts for
        this particular dataset, to be used when determine outputs
        from the model.

        Arguments:
            filepath: The filepath of the dataset.
            index_col: The index column in the dataset
            target_var_name: The target column in the data, i.e., 'y' for instance
            cat_features: The categorical features in the data
            num_features: The numeric features in the data
            remove_underscores: Whether to remove underscores from feature names
            store_to_conversation: Whether to store the dataset to the conversation.
            skip_prompts: whether to skip prompt generation.
        Returns:
            success: Returns success if completed and store_to_conversation is set to true. Otherwise,
                     returns the dataset.
        """
        app.logger.info(f"Loading dataset at path {filepath}...")

        # Read the dataset and get categorical and numerical features
        dataset, y_values, categorical, numeric = read_and_format_data(filepath,
                                                                       index_col,
                                                                       target_var_name,
                                                                       cat_features,
                                                                       num_features,
                                                                       remove_underscores)

        if store_to_conversation:

            # Store the dataset
            self.conversation.add_dataset(dataset, y_values, categorical, numeric)

            """# Set up the parser
            self.parser = Parser(cat_features=categorical,
                                 num_features=numeric,
                                 dataset=dataset,
                                 target=list(y_values))

            # Generate the available prompts
            # make sure to add the "incorrect" temporary feature
            # so we generate prompts for this
            self.prompts = Prompts(cat_features=categorical,
                                   num_features=numeric,
                                   target=np.unique(list(y_values)),
                                   feature_value_dict=self.parser.features,
                                   class_names=self.conversation.class_names,
                                   skip_creating_prompts=skip_prompts)"""
            app.logger.info("..done")

            return "success"
        else:
            return dataset, y_values

    def set_num_prompts(self, num_prompts):
        """Updates the number of prompts to a new number"""
        self.prompts.set_num_prompts(num_prompts)

    @staticmethod
    def gen_almost_surely_unique_id(n_bytes: int = 30):
        """To uniquely identify each input, we generate a random 30 byte hex string."""
        return secrets.token_hex(n_bytes)

    @staticmethod
    def log(logging_input: dict):
        """Performs the system logging."""
        assert isinstance(logging_input, dict), "Logging input must be dict"
        assert "time" not in logging_input, "Time field will be added to logging input"
        log_dialogue_input(logging_input)

    @staticmethod
    def build_logging_info(bot_name: str,
                           username: str,
                           response_id: str,
                           system_input: str,
                           parsed_text: str,
                           system_response: str):
        """Builds the logging dictionary."""
        return {
            'bot_name': bot_name,
            'username': username,
            'id': response_id,
            'system_input': system_input,
            'parsed_text': parsed_text,
            'system_response': system_response
        }

    def compute_parse_text(self, text: str, error_analysis: bool = False):
        """Computes the parsed text from the user text input.

        Arguments:
            error_analysis: Whether to do an error analysis step, where we compute if the
                            chosen prompts include all the
            text: The text the user provides to the system
        Returns:
            parse_tree: The parse tree from the formal grammar decoded from the user input.
            parse_text: The decoded text in the formal grammar decoded from the user input
                        (Note, this is just the tree in a string representation).
        """
        nn_prompts = None
        if error_analysis:
            grammar, prompted_text, nn_prompts = self.compute_grammar(text, error_analysis=error_analysis)
        else:
            grammar, prompted_text = self.compute_grammar(text, error_analysis=error_analysis)
        app.logger.info("About to decode")
        # Do guided-decoding to get the decoded text
        api_response = self.decoder.complete(
            prompted_text, grammar=grammar)
        decoded_text = api_response['generation']

        app.logger.info(f'Decoded text {decoded_text}')

        # Compute the parse tree from the decoded text
        # NOTE: currently, we're using only the decoded text and not the full
        # tree. If we need to support more complicated parses, we can change this.
        parse_tree, parsed_text = get_parse_tree(decoded_text)
        if error_analysis:
            return parse_tree, parsed_text, nn_prompts
        else:
            return parse_tree, parsed_text,

    def compute_parse_text_t5(self, text: str):
        """Computes the parsed text for the input using a t5 model.

        This supposes the user has finetuned a t5 model on their particular task and there isn't
        a need to do few shot
        """
        grammar, prompted_text = self.compute_grammar(text)
        decoded_text = self.decoder.complete(text, grammar)
        app.logger.info(f"t5 decoded text {decoded_text}")
        parse_tree, parse_text = get_parse_tree(decoded_text[0])
        return parse_tree, parse_text

    def compute_grammar(self, text, error_analysis: bool = False):
        """Computes the grammar from the text.

        Arguments:
            text: the input text
            error_analysis: whether to compute extra information used for error analyses
        Returns:
            grammar: the grammar generated for the input text
            prompted_text: the prompts computed for the input text
            nn_prompts: the knn prompts, without extra information that's added for the full
                        prompted_text provided to prompt based models.
        """
        nn_prompts = None
        app.logger.info("getting prompts")
        # Compute KNN prompts
        if error_analysis:
            prompted_text, adhoc, nn_prompts = self.prompts.get_prompts(text,
                                                                        self.prompt_metric,
                                                                        self.prompt_ordering,
                                                                        error_analysis=error_analysis)
        else:
            prompted_text, adhoc = self.prompts.get_prompts(text,
                                                            self.prompt_metric,
                                                            self.prompt_ordering,
                                                            error_analysis=error_analysis)
        app.logger.info("getting grammar")
        # Compute the formal grammar, making modifications for the current input
        grammar = self.parser.get_grammar(
            adhoc_grammar_updates=adhoc)

        if error_analysis:
            return grammar, prompted_text, nn_prompts
        else:
            return grammar, prompted_text

    def update_state(self, text: str, user_session_conversation: Conversation):
        """The main conversation driver.

        The function controls state updates of the conversation. It accepts the
        user input and ultimately returns the updates to the conversation.

        Arguments:
            text: The input from the user to the conversation.
            user_session_conversation: The conversation sessions for the current user.
        Returns:
            output: The response to the user input.
        """

        if any([text is None, self.prompts is None, self.parser is None]):
            return ''

        app.logger.info(f'USER INPUT: {text}')
        if False:  # "t5" not in self.decoding_model_name
            parse_tree, parsed_text = self.compute_parse_text(text)
        else:
            pass
            # parse_tree, parsed_text = self.compute_parse_text_t5(text) #We don't need text parsing for now.

        # Run the action in the conversation corresponding to the formal grammar
        if False:  # "t5" not in self.decoding_model_name
            returned_item = run_action(
                user_session_conversation, parse_tree, parsed_text)
        else:
            instance_id = self.current_instance[0]
            returned_item = run_action_by_id(user_session_conversation, int(text), instance_id)

        username = user_session_conversation.username

        response_id = self.gen_almost_surely_unique_id()
        """logging_info = self.build_logging_info(self.bot_name,
                                               username,
                                               response_id,
                                               text,
                                               parsed_text,
                                               returned_item)"""
        # self.log(logging_info) # Logging dict currently off.
        # Concatenate final response, parse, and conversation representation
        # This is done so that we can split both the parse and final
        # response, then present all the data
        final_result = returned_item + f"<>{response_id}"

        return final_result

    def update_state_dy_id(self,
                           question_id: int,
                           user_session_conversation: Conversation,
                           feature_id: int = None):
        """The main experiment driver.

                The function controls state updates of the conversation. It accepts the
                user input as question_id and feature_id and returns the updates to the conversation.

                Arguments:
                    question_id: The question id from the user.
                    feature_id: The feature id that the question is about.
                Returns:
                    output: The response to the user input.
                """

        if any([question_id is None]):
            return ''

        app.logger.info(f'USER INPUT: q_id:{question_id}, f_id:{feature_id}')
        instance_id = self.current_instance[0]
        returned_item = run_action_by_id(user_session_conversation, int(question_id), instance_id,
                                         int(feature_id))

        username = user_session_conversation.username  # TODO: Check if needed?!

        response_id = self.gen_almost_surely_unique_id()
        """logging_info = self.build_logging_info(self.bot_name,
                                               username,
                                               response_id,
                                               text,
                                               parsed_text,
                                               returned_item)"""  # TODO: Enable when logging works
        # self.log(logging_info) # Logging dict currently off.
        # Concatenate final response, parse, and conversation representation
        # This is done so that we can split both the parse and final
        # response, then present all the data
        # final_result = returned_item + f"<>{response_id}"
        final_result = returned_item
        return final_result

    def build_exit_survey_table(self):
        mega_explainer = self.conversation.get_var('mega_explainer').contents
        diverse_instances = self.conversation.get_var('diverse_instances').contents
        # Load md file
        file_loader = FileSystemLoader('.')
        env = Environment(loader=file_loader)
        template = env.get_template('templates/exit_questionnaire_template.md')
        model = self.conversation.get_var("model").contents
        exp_helper = self.conversation.get_var('experiment_helper').contents

        def get_features_by_avg_rank(lists):
            """
            Computes the feature(s) with the highest or lowest average rank across multiple lists.

            Parameters:
                lists (list[list[str]]): A list of lists, where each list contains a set of features in a specific order.


            Returns:
                Tuple[list[str], list[str]]: A Tuple of lists of features with the highest (Tuple[0]) and lowest (Tuple[1])
                 average rank across all lists.
            """

            # Create a dictionary to store the total rank of each feature and the number of times it appears in the lists
            feature_count = {}
            total_rank = {}

            # Loop through the lists and calculate the total rank of each feature
            for lst in lists:
                for i, feature in enumerate(lst):
                    feature_count[feature] = feature_count.get(feature, 0) + 1
                    total_rank[feature] = total_rank.get(feature, 0) + i + 1

            # Calculate the average rank for each feature
            avg_ranks = {feature: total_rank[feature] / feature_count[feature] for feature in feature_count}
            # Return all features ordered by the highest or lowest average rank
            return sorted(avg_ranks, key=avg_ranks.get, reverse=False), sorted(avg_ranks, key=avg_ranks.get,
                                                                               reverse=True)

        def turn_df_instance_to_dict(instance):
            """
            Change pandas instance to a dictionary to print to md file.
            """
            person_dict = {}
            for col_id, (key, value) in enumerate(instance.to_dict().items()):
                col_name = instance.columns[col_id]
                if col_name in self.categorical_features:
                    value = self.categorical_mapping[col_id][int(value[instance.index[0]])]
                else:
                    value = value[instance.index[0]]
                person_dict[key] = value
            return person_dict

        # First, get most important feature across all instances
        feature_importances_list = []
        # iterate over data df and handle each row as an instance (pandas df)
        for row in diverse_instances:
            data = pd.DataFrame(row['values'], index=[row['id']])
            feature_importance_dict = mega_explainer.get_feature_importances(data, [], False)[0]
            for label, feature_importances in feature_importance_dict.items():
                feature_importances_list.append(list(feature_importances.keys()))

        most_important_features_list, least_important_features_list = get_features_by_avg_rank(feature_importances_list)

        # Second, counterfactual thinking
        ### get a random instance from the dataset
        cf_count = 0
        for instance in diverse_instances:
            if cf_count == 2:
                break
            # turn the instance into a pandas df
            instance = pd.DataFrame(instance['values'], index=[instance['id']])
            instance_copy = instance.copy()
            # change slightly the attributes of the instance
            instance_copy = exp_helper.get_similar_instances(instance_copy, model, self.actionable_features)

            # Turn instance into key-value dict
            a2_instance_dict = turn_df_instance_to_dict(instance_copy)
            prediction = model.predict(instance_copy)[0]
            # Get necessary textes
            prediction_text = self.conversation.class_names[prediction]
            alternative_prediction_text = self.conversation.class_names[1 - prediction]

            # Find such cfe's that only a single attribute is changed.
            feature_names_to_value_mapping = {}
            for feature in instance.columns:
                cfe_string, _ = explain_cfe_by_given_features(self.conversation, instance, [feature])
                if cfe_string != 'There are no changes possible to the chosen attribute alone that would result in a different prediction.':
                    try:
                        feature_name = cfe_string.split("change")[1].split("to")[0].strip()
                        alternative_feature_value = cfe_string.split("change")[1].split("to")[1].split("<")[0].strip()
                    except IndexError:  # numerical feature
                        feature_name = cfe_string.split("crease")[1].split("to")[0].strip()  # increase or decrease
                        alternative_feature_value = cfe_string.split("crease")[1].split("to")[1].split("<")[0].strip()
                    feature_names_to_value_mapping[feature_name] = alternative_feature_value
                if len(feature_names_to_value_mapping) > 1 and cf_count == 0:
                    break  # Stop for first CF. Only need one.

            if len(feature_names_to_value_mapping) == 0:
                continue  # if such cfe's don't exist, continue with next instance
            if len(feature_names_to_value_mapping) < 3 and cf_count == 1:
                continue  # if second cf round, we need more possible cfs... at least 3!

            # Get textes for each feature change
            for feature_name, alternative_feature_value in feature_names_to_value_mapping.items():
                feature_names_to_value_mapping[feature_name] = alternative_feature_value
                feature_index = instance.columns.get_loc(feature_name)
                if feature_name in self.categorical_features:
                    alt_feature_values = self.categorical_mapping[feature_index].copy()
                    alt_feature_values.remove(alternative_feature_value)
                else:
                    # for numerical values...
                    # TODO: HOW TO HANDLE THIS?!
                    alt_feature_values = ["ADD_VALUE_HERE", "ADD_VALUE_HERE"]
                # if first cf, save it

                if cf_count == 0:
                    a2_q1_1 = a2_instance_dict
                    a2_q1_2 = prediction_text
                    a2_q1_3 = feature_name
                    a2_q1_4 = alternative_prediction_text
                    a2_q1_5 = f"Change {feature_name} to {alternative_feature_value}"
                    a2_q1_6 = f"Change {feature_name} to {alt_feature_values[0]}"
                    a2_q1_7 = f"Change {feature_name} to {alt_feature_values[1]}"
                    break
            # For second cf, save the 3 possibilities
            if cf_count == 1:
                # Get multiple counterfactuals that are true and make 2 up that are false.
                cf_string_list = []
                for feature, value in feature_names_to_value_mapping.items():
                    cf_string_list.append(f"Change {feature} to {value}")

                non_cf_string_list = []
                non_cf_features = list(set(instance.columns) - set(feature_names_to_value_mapping.keys()))
                for feature in non_cf_features:
                    # Value should be different from original instance
                    original_value = instance[feature].values[0]
                    feature_id = instance.columns.get_loc(feature)
                    if feature in self.categorical_features:
                        original_categorical = self.categorical_mapping[feature_id][
                            original_value]
                        non_cf_feature_values = self.categorical_mapping[feature_id].copy()
                        non_cf_feature_values.remove(original_categorical)
                        non_cf_string_list.append(f"Change {feature} to {non_cf_feature_values[-1]}")
                    else:
                        non_cf_string_list.append(f"Change {feature} to ADD_VALUE_HERE")

                a2_q2_1 = a2_instance_dict
                a2_q2_2 = prediction_text
                a2_q2_3 = alternative_prediction_text
                a2_q2_4 = cf_string_list[0]
                a2_q2_5 = cf_string_list[1]
                a2_q2_6 = non_cf_string_list[0]
                a2_q2_7 = non_cf_string_list[1]
                a2_q2_8 = cf_string_list[2]

            cf_count += 1

        # Third, Simulate Model Behavior
        # 3.1 Present an instance and ask for prediction
        for instance in diverse_instances:
            # turn the instance into a pandas df
            instance = pd.DataFrame(instance['values'], index=[instance['id']])
            instance_copy = instance.copy()

            # Change some attributes that don't change the prediction
            instance_copy = exp_helper.get_similar_instances(instance_copy, model, self.changeable_features)
            if instance_copy is None:
                continue

            # turn instance_copy into a dict
            a3_1_instance_dict = turn_df_instance_to_dict(instance_copy)

        # 3.2 present 3 instances and ask which is most likely to be high risk
        found_high_risk = None
        not_high_risk = []
        for instance in diverse_instances:
            # turn the instance into a pandas df
            instance = pd.DataFrame(instance['values'], index=[instance['id']])
            instance_copy = instance.copy()

            # Change some attributes that don't change the prediction
            instance_copy = exp_helper.get_similar_instances(instance_copy, model, self.changeable_features)
            if instance_copy is None:
                continue

            # get instance prediction
            instance_prediction = model.predict(instance_copy)[0]
            if instance_prediction == 0:
                found_high_risk = turn_df_instance_to_dict(instance_copy)
            else:
                not_high_risk.append(turn_df_instance_to_dict(instance_copy))
            if len(not_high_risk) > 1 and found_high_risk is not None:
                break

        markdown = template.render(
            a1_q1_1=most_important_features_list[0],
            a1_q1_2=least_important_features_list[1],
            a1_q1_3=least_important_features_list[2],
            a1_q2_1=least_important_features_list[0],
            a1_q2_2=most_important_features_list[1],
            a1_q2_3=most_important_features_list[2],
            a2_q1_1=a2_q1_1,
            a2_q1_2=a2_q1_2,
            a2_q1_3=a2_q1_3,
            a2_q1_4=a2_q1_4,
            a2_q1_5=a2_q1_5,
            a2_q1_6=a2_q1_6,
            a2_q1_7=a2_q1_7,
            a2_q2_1=a2_q2_1,
            a2_q2_2=a2_q2_2,
            a2_q2_3=a2_q2_3,
            a2_q2_4=a2_q2_4,
            a2_q2_5=a2_q2_5,
            a2_q2_6=a2_q2_6,
            a2_q2_7=a2_q2_7,
            a2_q2_8=a2_q2_8,
            a3_q1_1=a3_1_instance_dict,
            a3_q1_2=prediction_text,
            a3_q1_3=alternative_prediction_text,
            a3_q2_1=not_high_risk[0],
            a3_q2_2=not_high_risk[1],
            a3_q2_3=found_high_risk,
            a3_q2_4=self.conversation.class_names[0]
        )

        # Save the rendered Markdown to a file
        output_file = '03_exit_questionnaire_filled.md'
        with open(output_file, 'w') as file:
            file.write(markdown)
