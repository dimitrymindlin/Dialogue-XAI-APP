"""The main script that controls conversation logic.

This file contains the core logic for facilitating conversations. It orchestrates the necessary
routines for setting up conversations, controlling the state of the conversation, and running
the functions to get the responses to user inputs.
"""
import asyncio
import difflib
import json
import os
import pickle
from random import seed as py_random_seed
from typing import List, Tuple, Optional, Dict, Any
import re

from jinja2 import Environment, FileSystemLoader
import numpy as np
import pandas as pd

from flask import Flask
import gin

from create_experiment_data.experiment_helper import ExperimentHelper
from create_experiment_data.instance_datapoint import InstanceDatapoint
from data.response_templates.template_manager import TemplateManager
from create_experiment_data.test_instances import TestInstances
from explain.action import run_action, run_action_new, compute_explanation_report
from explain.actions.explanation import explain_cfe_by_given_features
from explain.actions.static_followup_options import get_mapping
from explain.conversation import Conversation
from explain.dialogue_manager.manager import DialogueManager
from explain.explanation import MegaExplainer
from explain.explanations.anchor_explainer import TabularAnchor
from explain.explanations.ceteris_paribus import CeterisParibus
from explain.explanations.dice_explainer import TabularDice
from explain.explanations.diverse_instances import DiverseInstances
from explain.explanations.feature_statistics_explainer import FeatureStatisticsExplainer
from explain.explanations.model_profile import PdpExplanation
from explain.utils import read_and_format_data
from explain.xai_cache_manager import XAICacheManager

from parsing.llm_intent_recognition.llm_pipeline_setup.openai_pipeline.openai_pipeline import \
    LLMSinglePromptWithMemoryAndSystemMessage

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
                 study_group: str,
                 ml_knowledge: str,
                 user_id: str,
                 model_file_path: str,
                 dataset_file_path: str,
                 background_dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: List[str],
                 ordinary_features: List[str],
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
                 instance_type_naming: str = "instance",
                 encoded_col_mapping_path: dict = None,
                 feature_name_mapping_path: str = None,
                 use_selection: bool = False,
                 use_intent_recognition: bool = False,
                 use_active_dialogue_manager: bool = False,
                 use_llm_agent=False,
                 use_static_followup=False,
                 use_two_prompts=False,
                 submodular_pick: bool = False):
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
            ordinary_features: The names of the ordinal (categorical) features in the data.
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
            actionable_features: A list of features that can be changed (actionable features)
            feature_units_mapping: A mapping from feature names to units. This is used to display units in the UI.
            instance_type_naming: The naming of the instance type. This is used to display the instance type such as
                                    "person" or "house" in the UI.
            encoded_col_mapping_path: Path to the encoded column mapping file.
            feature_name_mapping_path: Path to the feature name mapping file.
        """

        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)

        self.bot_name = name
        self.study_group = study_group
        self.ml_knowledge = ml_knowledge

        # Prompt settings
        self.prompt_metric = prompt_metric
        self.prompt_ordering = prompt_ordering
        self.use_guided_decoding = use_guided_decoding
        self.categorical_features = categorical_features
        self.ordinary_features = ordinary_features
        self.numerical_features = numerical_features
        self.instance_type_naming = instance_type_naming
        self.encoded_col_mapping_path = encoded_col_mapping_path
        self.feature_name_mapping_path = feature_name_mapping_path
        self.feature_ordering = None
        self.submodular_pick = submodular_pick
        self.use_selection = use_selection
        self.use_intent_recognition = use_intent_recognition
        self.use_active_dialogue_manager = use_active_dialogue_manager

        # Check environment variable first, fallback to gin parameter
        env_llm_agent = os.getenv('XAI_USE_LLM_AGENT')
        if env_llm_agent is not None:
            # Convert string "False" to boolean False
            if env_llm_agent.lower() == 'false':
                self.use_llm_agent = False
            else:
                self.use_llm_agent = env_llm_agent
            print(f"Using LLM agent from environment variable: {self.use_llm_agent}")
        else:
            self.use_llm_agent = use_llm_agent
            print(f"Using LLM agent from gin configuration: {self.use_llm_agent}")

        self.use_static_followup = use_static_followup
        self.use_two_prompts = use_two_prompts
        if self.use_static_followup:
            self.static_followup_mapping = get_mapping()

        # A variable used to help file uploads
        self.manual_var_filename = None

        self.decoding_model_name = parsing_model_name

        self.decoder = None

        self.data_instances = []
        self.train_instance_counter = 0
        self.test_instance_counter = 0
        self.user_prediction_dict = {}
        self.current_instance: InstanceDatapoint = None
        self.current_instance_type = "train"  # Or test

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
                self.categorical_mapping = {str(k): v for k, v in categorical_mapping.items()}
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

        # Load Template Manager
        template_manager = TemplateManager(self.conversation,
                                           encoded_col_mapping_path=encoded_col_mapping_path,
                                           categorical_mapping=categorical_mapping,
                                           feature_name_mapping_path=self.feature_name_mapping_path)
        self.conversation.add_var('template_manager', template_manager, 'template_manager')
        self.feature_ordering = list(template_manager.feature_display_names.feature_name_to_display_name.keys())

        # Load Experiment Helper
        helper = ExperimentHelper(conversation=self.conversation,
                                  categorical_mapping=self.categorical_mapping,
                                  categorical_features=self.categorical_features,
                                  template_manager=template_manager)
        self.conversation.add_var('experiment_helper', helper, 'experiment_helper')

        # Initialize completion + parsing modules
        self.intent_recognition_model = None
        if self.use_intent_recognition == "openAI":
            self.intent_recognition_model = LLMSinglePromptWithMemoryAndSystemMessage(self.feature_ordering)

        if self.use_llm_agent:
            if self.use_llm_agent == "o1":
                from llm_agents.o1_agent.openai_o1_agent import XAITutorAssistant as Agent
            elif self.use_llm_agent in ("mape_k", "mape_k_4"):
                from llm_agents.mape_k_mixins import MapeK4BaseAgent as Agent
            elif self.use_llm_agent == "mape_k_2":
                from llm_agents.mape_k_mixins import MapeK2BaseAgent as Agent
            elif self.use_llm_agent in ("unified_mape_k", "mape_k_unified"):
                from llm_agents.mape_k_mixins import MapeKUnifiedBaseAgent as Agent
            elif self.use_llm_agent == "mape_k_approval_2":
                from llm_agents.mape_k_mixins import MapeKApprovalBaseAgent as Agent
            elif self.use_llm_agent in ("mape_k_approval", "mape_k_approval_4"):
                from llm_agents.mape_k_mixins import MapeKApproval4BaseAgent as Agent
            elif self.use_llm_agent == "mape_k_openai":
                from llm_agents.openai_mapek_agent import MapeK4OpenAIAgent as Agent
            elif self.use_llm_agent == "mape_k_openai_2":
                from llm_agents.openai_mapek_agent import MapeK2OpenAIAgent as Agent
            elif self.use_llm_agent == "mape_k_openai_unified":
                from llm_agents.openai_mapek_agent import MapeKUnifiedOpenAIAgent as Agent
            else:
                raise ValueError(f"Unknown agent type: {self.use_llm_agent}")
            self.agent = Agent(
                feature_names=self.get_feature_names(),
                feature_units=self.get_feature_units(),
                feature_tooltips=self.get_feature_tooltips(),
                domain_description=self.conversation.describe.get_dataset_description(),
                user_ml_knowledge=self.ml_knowledge,
                experiment_id=user_id  # Pass user_id as initial experiment_id
            )

        # Load the explanations
        self.load_explanations(background_ds_x=background_dataset,
                               background_ds_y=background_y_values)

        ## Initialize Dialogue Manager
        self.dialogue_manager = DialogueManager(intent_recognition=self.intent_recognition_model,
                                                template_manager=template_manager,
                                                active=self.use_active_dialogue_manager)

        # Initialize XAI Cache Manager
        self.xai_cache_manager = XAICacheManager()

    def get_feature_display_name_dict(self):
        template_manager = self.conversation.get_var('template_manager').contents
        return template_manager.feature_display_names.feature_name_to_display_name

    def get_feature_ranges(self):
        feature_statistics_explainer = self.conversation.get_var('feature_statistics_explainer').contents
        return feature_statistics_explainer.get_feature_ranges()

    def set_user_prediction(self, experiment_phase, datapoint_count, user_prediction):
        reversed_dict = {v: k for k, v in self.conversation.class_names.items()}
        user_prediction_as_int = reversed_dict.get(user_prediction, 1000)  # 1000 is for "I don't know" option
        
        # Check if the experiment_phase exists in user_prediction_dict
        if experiment_phase not in self.user_prediction_dict:
            raise ValueError(f"Experiment phase '{experiment_phase}' not found. You must request a datapoint before setting a prediction.")
        
        # Check if the datapoint_count exists for this experiment_phase
        if datapoint_count not in self.user_prediction_dict[experiment_phase]:
            raise ValueError(f"Datapoint {datapoint_count} not found for phase '{experiment_phase}'. You must request this specific datapoint before setting a prediction.")
        
        entry = self.user_prediction_dict[experiment_phase][datapoint_count]
        entry['user_prediction'] = user_prediction_as_int
        correct_pred = entry['true_label']
        return user_prediction_as_int == correct_pred, self.conversation.class_names[correct_pred]

    def get_user_correctness(self, train=False):
        predictions = self.user_prediction_dict["train" if train else "test"]
        total = len(predictions)
        correct = sum(1 for p in predictions.values() if p["user_prediction"] == p["true_label"])
        return f"{correct} out of {total}"

    def get_proceeding_okay(self):
        return self.dialogue_manager.get_proceeding_okay()

    def get_next_instance(self, instance_type, datapoint_count, return_probability=False) -> InstanceDatapoint:
        """
        Returns the next instance in the data_instances list if possible.
        param instance_type: type of instance to return, can be train, test or final_test
        """
        self.dialogue_manager.reset_state()
        experiment_helper = self.conversation.get_var('experiment_helper').contents
        self.current_instance = experiment_helper.get_next_instance(
            instance_type=instance_type,
            datapoint_count=datapoint_count,
            return_probability=return_probability)
        # Update agent with new instance
        if self.use_llm_agent and instance_type == "train":
            # Try to get precomputed XAI data from cache first
            cached_data = self.xai_cache_manager.get_cached_xai_report(self.current_instance.instance_id)

            if cached_data and cached_data.get("is_valid", False):
                # Use cached data for instant response
                print(f"Using cached XAI data for instance {self.current_instance.instance_id}")
                xai_report = cached_data["xai_report"]
                visual_exp_dict = cached_data.get("visual_explanations", {})
                if "FeatureInfluencesPlot" not in visual_exp_dict:
                    visual_exp_dict["FeatureInfluencesPlot"] = self.update_state_new(question_id="shapAllFeaturesPlot")[
                        0]
            else:
                # Fallback to synchronous computation with loading indication
                print(f"Cache miss for instance {self.current_instance.instance_id}, computing synchronously...")
                xai_report = self.get_explanation_report(as_text=True)
                # Get visual explanations
                visual_exp_dict = {}
                visual_exp_dict["FeatureInfluencesPlot"] = self.update_state_new(question_id="shapAllFeaturesPlot")[0]

                # Cache the computed results for future use
                self.xai_cache_manager.compute_and_cache_xai_report(
                    self.conversation,
                    self.current_instance.instance_id,
                    self.get_feature_display_name_dict()
                )

            opposite_class_name = self.conversation.class_names[1 - self.get_current_prediction(as_int=True)]
            self.agent.initialize_new_datapoint(self.current_instance, xai_report, visual_exp_dict,
                                                self.get_current_prediction(),
                                                opposite_class_name=opposite_class_name,
                                                datapoint_count=datapoint_count)
        # Update user_prediction_dict with current instance's ML prediction
        ml_prediction = self.get_current_prediction(as_int=True)
        try:
            self.user_prediction_dict[instance_type][self.current_instance.counter] = {"true_label": ml_prediction}
        except KeyError:
            self.user_prediction_dict[instance_type] = {self.current_instance.counter: {"true_label": ml_prediction}}
        return self.current_instance

    def get_study_group(self):
        return self.study_group

    def get_current_prediction(self, as_int=False):
        """
        Returns the current prediction.
        """
        if as_int:
            return self.current_instance.model_predicted_label
        return self.current_instance.model_predicted_label_string

    def get_feature_tooltips(self):
        """
        Returns the feature tooltips for the current dataset.
        """
        template_manager = self.conversation.get_var("template_manager").contents
        return template_manager.feature_display_names.feature_tooltips

    def get_feature_units(self):
        """
        Returns the feature units for the current dataset.
        """
        template_manager = self.conversation.get_var("template_manager").contents
        return template_manager.feature_display_names.feature_units

    def generate_baseline_probability_text(self) -> str:
        """
        Generate dataset-dependent baseline probability text for user feedback.
        
        Returns:
            String describing the baseline probability and model approach
        """
        class_names = self.conversation.class_names

        if not class_names:
            return "The model looks at the person's information to make a prediction."

        # Try to get actual SHAP base value from the conversation
        base_value = None
        try:
            mega_explainer = self.conversation.get_var('mega_explainer').contents
            if 'shap' in mega_explainer.mega_explainer.explanation_methods:
                shap_explainer = mega_explainer.mega_explainer.explanation_methods['shap']
                base_value = shap_explainer.feature_explainer.expected_value[0]
            elif hasattr(mega_explainer.mega_explainer,
                         'explanation_methods') and mega_explainer.mega_explainer.explanation_methods:
                # Try to get base value from any available SHAP explainer
                for method_name, explainer in mega_explainer.mega_explainer.explanation_methods.items():
                    if hasattr(explainer, 'explainer') and hasattr(explainer.feature_explainer, 'expected_value'):
                        base_value = explainer.feature_explainer.expected_value[0]
                        break
        except (AttributeError, KeyError, IndexError):
            base_value = None

        # Extract class labels
        class_labels = list(class_names.values())

        # If we have a base value, use it; otherwise use a generic message
        if base_value is not None:
            base_percentage = round(base_value * 100)

            # Determine which class the base value represents
            # If base_value > 0.5, it favors the positive class (class_labels[1])
            # If base_value <= 0.5, it favors the negative class (class_labels[0])
            if base_value > 0.5:
                baseline_class = class_labels[1] if len(class_labels) > 1 else class_labels[0]
            else:
                baseline_class = class_labels[0]

            return f"The model starts by assuming a <b>{base_percentage}% chance</b> that someone is <b>{baseline_class}</b>, then looks at this person's specific information to adjust that prediction."
        else:
            # Use the first class (typically negative class) as baseline for generic message
            baseline_class = class_labels[0]
            return f"The model looks at this person's information to decide if they are more likely to be <b>{baseline_class}</b> or <b>{class_labels[1] if len(class_labels) > 1 else 'the other class'}</b>."

    def get_feature_names(self):
        template_manager = self.conversation.get_var("template_manager").contents
        experiment_helper = self.conversation.get_var("experiment_helper").contents
        feature_display_names = template_manager.feature_display_names.feature_name_to_display_name
        feature_names = list(self.conversation.get_var("dataset").contents['X'].columns)
        original_feature_names = list(self.conversation.get_var("dataset").contents['X'].columns)

        # Use experiment helper's feature ordering which preserves config order
        if hasattr(experiment_helper, 'feature_ordering') and experiment_helper.feature_ordering:
            # Convert display names back to feature names and maintain order
            display_name_to_feature_name = {v: k for k, v in feature_display_names.items()}
            ordered_feature_names = []

            for display_name in experiment_helper.feature_ordering:
                # Find the corresponding feature name (original column name)
                feature_name = display_name_to_feature_name.get(display_name, display_name)
                # Handle cases where display name doesn't match exactly - try without spaces
                if feature_name not in feature_names:
                    feature_name = display_name.replace(" ", "")
                if feature_name in feature_names:
                    ordered_feature_names.append(feature_name)

            # Add any remaining feature names that weren't in the ordering
            for feature_name in feature_names:
                if feature_name not in ordered_feature_names:
                    ordered_feature_names.append(feature_name)

            feature_names = ordered_feature_names
        else:
            # Fallback: use existing ordering or sort alphabetically
            if self.feature_ordering is not None:
                # Sort feature names by feature_ordering
                feature_names_ordering = [feature.replace(" ", "") for feature in
                                          self.feature_ordering]  # From display names to feature names
                feature_names = sorted(feature_names, key=lambda k: feature_names_ordering.index(
                    k) if k in feature_names_ordering else len(feature_names_ordering))
            else:
                feature_names = sorted(feature_names)

        # Map feature names to their original IDs and display names, if available
        feature_names_id_mapping = [
            {'id': original_feature_names.index(feature_name),
             'feature_name': feature_display_names.get(feature_name, feature_name)}
            for feature_name in feature_names
        ]

        return feature_names_id_mapping

    def init_loaded_var(self, name: bytes):
        """Inits a var from manual load."""
        self.manual_var_filename = name.decode("utf-8")

    def get_questions_attributes_featureNames(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the questions and attributes and feature names for the current dataset.
        """
        try:
            # Read the question bank CSV file
            question_pd = pd.read_csv(self.conversation.question_bank_path, delimiter=";")

            # Replace "instance" in all 'paraphrased' entries with instance_type_naming
            question_pd["paraphrased"] = question_pd["paraphrased"].str.replace("instance", self.instance_type_naming)

            # Create answer dictionary with general and feature questions
            answer_dict = {
                "general_questions": question_pd[question_pd["question_type"] == "general"]
                                     .loc[:, ['q_id', 'paraphrased']]
                .rename(columns={'paraphrased': 'question'})
                .to_dict(orient='records'),

                "feature_questions": question_pd[question_pd["question_type"] == "feature"]
                                     .loc[:, ['q_id', 'paraphrased']]
                .rename(columns={'paraphrased': 'question'})
                .to_dict(orient='records')
            }

            return answer_dict

        except FileNotFoundError:
            raise Exception(f"File not found: {self.conversation.question_bank_path}")
        except pd.errors.EmptyDataError:
            raise Exception("The question bank CSV file is empty or invalid.")

    def load_explanations(self,
                          background_ds_x,
                          background_ds_y=None):
        """Loads the explanations.

        If set in gin, this routine will cache the explanations.

        Arguments:
            background_ds_x: The background dataset to compute the explanations with.
            background_ds_y: The background dataset's y values.
        """
        app.logger.info("Loading explanations into conversation...")

        # This may need to change as we add different types of models
        pred_f = self.conversation.get_var('model_prob_predict').contents
        model = self.conversation.get_var('model').contents
        test_data = self.conversation.get_var('dataset').contents['X']
        test_data_y = self.conversation.get_var('dataset').contents['y']
        categorical_f = self.conversation.get_var('dataset').contents['cat']
        numeric_f = self.conversation.get_var('dataset').contents['numeric']

        # Load local FI explanations
        app.logger.info("...loading MegaExplainer...")
        mega_explainer = MegaExplainer(model=model,
                                       prediction_fn=pred_f,
                                       data=background_ds_x,
                                       cat_features=categorical_f,
                                       class_names=self.conversation.class_names,
                                       categorical_mapping=self.categorical_mapping)

        # Load diverse instances (explanations)
        app.logger.info("...loading DiverseInstances...")

        if self.submodular_pick:
            feature_explainer_for_sp = mega_explainer.mega_explainer.explanation_methods['shap']
        else:
            feature_explainer_for_sp = None
        diverse_instances_explainer = DiverseInstances(
            dataset_name=self.conversation.describe.dataset_name,
            feature_explainer=feature_explainer_for_sp)
        diverse_instance_ids = diverse_instances_explainer.get_instance_ids_to_show(data=test_data,
                                                                                    model=model,
                                                                                    y_values=test_data_y,
                                                                                    submodular_pick=self.submodular_pick)
        # Make new list of dicts {id: instance_dict} where instance_dict is a dict with column names as key and values as values.
        diverse_instances = [{"id": i, "values": test_data.loc[i].to_dict()} for i in diverse_instance_ids]
        diverse_instance_ids = [d['id'] for d in diverse_instances]
        app.logger.info(f"...loaded {len(diverse_instance_ids)} diverse instance ids from cache!")

        # Compute explanations for diverse instances

        ## Load dice explanations
        tabular_dice = TabularDice(model=model,
                                   data=test_data,
                                   num_features=numeric_f,
                                   class_names=self.conversation.class_names,
                                   categorical_mapping=self.categorical_mapping,
                                   background_dataset=background_ds_x,
                                   features_to_vary=self.conversation.get_var(
                                       "experiment_helper").contents.actionable_features)
        tabular_dice.get_explanations(ids=diverse_instance_ids,
                                      data=test_data)

        # Remove ids without cfes from diverse_instance_ids
        if tabular_dice.ids_without_cfes:
            diverse_instances = [d for d in diverse_instances if d['id'] not in tabular_dice.ids_without_cfes]
            diverse_instance_ids = [d['id'] for d in diverse_instances]

        message = f"...loaded {len(tabular_dice.cache)} dice tabular explanations from cache!"
        app.logger.info(message)

        ## Feature Importance
        mega_explainer.get_explanations(ids=diverse_instance_ids, data=test_data)
        message = f"...loaded {len(mega_explainer.cache)} mega explainer explanations from cache!"
        app.logger.info(message)

        # Load anchor explanations
        # categorical_names = create_feature_values_mapping_from_df(data, categorical_f)
        tabular_anchor = TabularAnchor(model=model,
                                       data=test_data,
                                       categorical_mapping=self.categorical_mapping,
                                       class_names=self.conversation.class_names,
                                       feature_names=list(test_data.columns))
        tabular_anchor.get_explanations(ids=diverse_instance_ids,
                                        data=test_data)

        # Load Ceteris Paribus Explanations
        ceteris_paribus_explainer = CeterisParibus(model=model,
                                                   background_data=background_ds_x,
                                                   ys=background_ds_y,
                                                   class_names=self.conversation.class_names,
                                                   feature_names=list(test_data.columns),
                                                   categorical_mapping=self.categorical_mapping,
                                                   ordinal_features=self.ordinary_features)
        ceteris_paribus_explainer.get_explanations(ids=diverse_instance_ids,
                                                   data=test_data)

        # Load global explanation via shap explainer
        # Create background_data from x and y dfs
        """shap_explainer = ShapGlobalExplainer(model=model,
                                             data=background_ds_x,
                                             class_names=self.conversation.class_names)

        shap_explainer.get_explanations()
        self.conversation.add_var('global_shap', shap_explainer, 'explanation')"""

        pdp_explainer = PdpExplanation(model=model,
                                       background_data=background_ds_x,
                                       ys=background_ds_y,
                                       feature_names=list(test_data.columns),
                                       categorical_features=self.categorical_features,
                                       numerical_features=self.numerical_features,
                                       categorical_mapping=self.categorical_mapping,
                                       dataset_name=self.conversation.describe.dataset_name)
        pdp_explainer.get_explanations()
        self.conversation.add_var('pdp', pdp_explainer, 'explanation')

        # Load FeatureStatisticsExplainer with background data
        feature_statistics_explainer = FeatureStatisticsExplainer(background_ds_x,
                                                                  background_ds_y,
                                                                  self.numerical_features,
                                                                  feature_names=list(background_ds_x.columns),
                                                                  rounding_precision=self.conversation.rounding_precision,
                                                                  categorical_mapping=self.categorical_mapping,
                                                                  feature_units=self.get_feature_units())
        self.conversation.add_var('feature_statistics_explainer', feature_statistics_explainer, 'explanation')

        # Add all the explanations to the conversation
        self.conversation.add_var('mega_explainer', mega_explainer, 'explanation')
        self.conversation.add_var('tabular_dice', tabular_dice, 'explanation')
        self.conversation.add_var('tabular_anchor', tabular_anchor, 'explanation')
        self.conversation.add_var('ceteris_paribus', ceteris_paribus_explainer, 'explanation')
        # list of dicts {id: instance_dict} where instance_dict is a dict with column names as key and values as values.
        # Load test instances
        test_instance_explainer = TestInstances(test_data, model,
                                                self.conversation.get_var("experiment_helper").contents,
                                                diverse_instance_ids=diverse_instance_ids,
                                                actionable_features=self.conversation.get_var(
                                                    "experiment_helper").contents.actionable_features,
                                                categorical_features=self.categorical_features, )

        test_instances, final_test_instances = test_instance_explainer.get_test_instances()
        # given the list of remove_instances_from_experiment, remove them from the experiment in all explanations
        self.conversation.add_var('test_instances', test_instances, 'test_instances')
        self.conversation.add_var('diverse_instances', diverse_instances, 'diverse_instances')
        self.conversation.add_var('final_test_instances', final_test_instances, 'final_test_instances')

        # Save the cluster-based diverse instances (not the flattened list format)
        diverse_instances_explainer.save_diverse_instances(diverse_instances_explainer.diverse_instances)

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

        self.numerical_features = numeric
        self.categorical_features = categorical

        if store_to_conversation:

            # Store the dataset
            self.conversation.add_dataset(dataset, y_values, categorical, numeric)
            app.logger.info("..done")

            return "success"
        else:
            return dataset, y_values

    def get_suggested_method(self):
        return self.dialogue_manager.get_suggested_explanations()

    def get_static_followup(self, question_id) -> List[Dict[str, Any]]:
        # return example [{"id": "shapAllFeatures", "question": "Would you like to see the feature contributions?", "feature_id": None}]
        try:
            method_id, question = self.static_followup_mapping[question_id]
            return [{"question_id": method_id, "question": question, "feature_id": ""}]
        except (KeyError, TypeError):
            return []

    def update_state_new(self,
                         question_id: str = None,
                         feature_id: int = None) -> tuple[str, int, Optional[int]]:
        """The main experiment driver.

                The function controls state updates of the conversation. It accepts the
                user input as question_id and feature_id and returns the updates to the conversation.

                Arguments:
                    question_id: The question id from the user.
                    feature_id: The feature id that the question is about.
                Returns:
                    output: The response to the user input.
                """

        instance_id = self.current_instance.instance_id
        question_id, feature_name, reasoning = self.dialogue_manager.update_state(None, question_id, feature_id)

        if question_id is None:
            return '', None, None, reasoning

        if feature_id is not None and feature_id != "":
            feature_id = int(feature_id)

        app.logger.info(f'USER INPUT: q_id:{question_id}, f_id:{feature_id}')
        # Convert feature_id to int if not None
        returned_item = run_action_new(self.conversation,
                                       question_id,
                                       instance_id,
                                       feature_id,
                                       instance_type_naming=self.instance_type_naming)
        final_result = returned_item
        return final_result, question_id, feature_id, reasoning

    async def update_state_from_nl(self, user_input):
        # 1. Get the question_id and feature_name from the user input
        feature_name = None
        feature_id = None
        if self.use_llm_agent:
            reasoning, response = await self.agent.answer_user_question(user_input)
            return response, None, None, reasoning
        elif self.use_intent_recognition:
            # Get the question_id and feature_name from the user input
            question_id, feature_name, reasoning = self.dialogue_manager.update_state(user_input)
            if feature_name != "" and feature_name is not None:
                feature_list = [col.lower() for col in self.conversation.stored_vars['dataset'].contents['X'].columns]
                # remove whitespace between words
                feature_name = feature_name.replace(" ", "")
                try:
                    feature_id = feature_list.index(feature_name.lower())
                except ValueError:
                    # Get closest match
                    closest_matches = difflib.get_close_matches(feature_name, feature_list, n=1, cutoff=0.5)
                    if closest_matches:
                        feature_id = feature_list.index(closest_matches[0])
                    else:
                        feature_id = None
                        # Optionally handle the case where no close match is found
                        print(f"No close match found for feature name: {feature_name}")

        # 2. Update the state
        return self.update_state_new(question_id, feature_id)

    async def update_state_from_nl_stream(self, user_input):
        """
        Streaming version of update_state_from_nl.
        
        Args:
            user_input: User's natural language input
            
        Yields:
            Streaming chunks from the agent if streaming is supported, 
            otherwise yields the final result.
        """
        feature_name = None
        feature_id = None

        if self.use_llm_agent:
            # Check if agent supports streaming
            if hasattr(self.agent, 'answer_user_question_stream'):
                # Use streaming
                async for chunk in self.agent.answer_user_question_stream(user_input):
                    yield chunk
            else:
                # Fallback to normal response
                reasoning, response = await self.agent.answer_user_question(user_input)
                yield {
                    "type": "final",
                    "content": response,
                    "reasoning": reasoning,
                    "is_complete": True
                }
        elif self.use_intent_recognition:
            # Intent recognition doesn't support streaming, use normal flow
            question_id, feature_name, reasoning = self.dialogue_manager.update_state(user_input)
            if feature_name != "" and feature_name is not None:
                feature_list = [col.lower() for col in self.conversation.stored_vars['dataset'].contents['X'].columns]
                # remove whitespace between words
                feature_name = feature_name.replace(" ", "")
                try:
                    feature_id = feature_list.index(feature_name.lower())
                except ValueError:
                    # Get closest match
                    closest_matches = difflib.get_close_matches(feature_name, feature_list, n=1, cutoff=0.5)
                    if closest_matches:
                        feature_id = feature_list.index(closest_matches[0])
                    else:
                        feature_id = None
                        # Optionally handle the case where no close match is found
                        print(f"No close match found for feature name: {feature_name}")

            # Update the state and yield final result
            response, question_id, feature_id, reasoning = self.update_state_new(question_id, feature_id)
            yield {
                "type": "final",
                "content": response,
                "reasoning": reasoning,
                "question_id": question_id,
                "feature_id": feature_id,
                "is_complete": True
            }

    def get_feature_importances_for_current_instance(self):
        mega_explainer = self.conversation.get_var('mega_explainer').contents
        data = pd.DataFrame(self.current_instance.instance_as_dict, index=[self.current_instance.instance_id])
        feature_importance_dict = mega_explainer.get_feature_importances(data, [], False)[0]
        # Turn display names into feature names
        feature_importance_dict = {self.get_feature_display_name_dict().get(k, k): v for k, v in
                                   feature_importance_dict.items()}
        # Extract the feature importances for the current instance from outer dict with current class as key
        try:
            feature_importance_dict = feature_importance_dict[self.current_instance.model_predicted_label_string]
        except KeyError:
            # Get by int label
            feature_importance_dict = feature_importance_dict[self.current_instance.model_predicted_label]

        return feature_importance_dict

    def reset_dialogue_manager(self):
        """
        Resets the dialogue manager to the initial state and resets the feature importances for the current instance.
        """
        current_feature_importances = self.get_feature_importances_for_current_instance()
        self.dialogue_manager.reset_state()
        self.dialogue_manager.feature_importances = current_feature_importances

    def get_explanation_report(self, as_text=False):
        """Returns the explanation report."""
        instance_id = self.current_instance.instance_id
        report = compute_explanation_report(self.conversation,
                                            instance_id,
                                            instance_type_naming=self.instance_type_naming,
                                            feature_display_name_mapping=self.get_feature_display_name_dict(),
                                            as_text=as_text)
        return report

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
            instance_copy = exp_helper.get_similar_instance(instance_copy, model)

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
                        feature_name = cfe_string.split("Changing")[1].split("to")[0].strip()
                        alternative_feature_value = cfe_string.split("Changing")[1].split("to")[1].split("</em>")[0]
                    except IndexError:  # numerical feature
                        feature_name = cfe_string.split("creasing")[1].split("to")[0].strip()  # increase or decrease
                        alternative_feature_value = cfe_string.split("creasing")[1].split("to")[1].split("</em>")[0]
                    # remove html tags
                    alternative_feature_value = re.sub('<[^<]+?>', '', alternative_feature_value).strip()
                    feature_name = re.sub('<[^<]+?>', '', feature_name).strip()
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
            instance_copy = exp_helper.get_similar_instance(instance_copy, model, self.changeable_features)
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
            instance_copy = exp_helper.get_similar_instance(instance_copy, model, self.changeable_features)
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

    def save_all_questions_and_answers_to_csv(self, output_path="all_questions_and_answers.csv"):
        """Saves all possible question and answer combinations to a CSV file based on an intent DataFrame.

        This function iterates through the provided intent DataFrame, which contains columns for questions,
        XAI methods, and features. For each row, it calls the update_state_new method to get the answer,
        and saves the results to a CSV file.

        Arguments:
            intent_pd: Pandas DataFrame containing 'question', 'xai method', and 'feature' columns
            output_path: Path where to save the CSV file. Defaults to "all_questions_and_answers.csv".
        Returns:
            DataFrame: DataFrame with the original columns plus an 'answer' column.
        """
        import pandas as pd
        from parsing.llm_intent_recognition.prompts.explanations_prompt import question_to_id_mapping

        # Create a copy of the intent DataFrame to add the answers
        intent_pd = pd.read_csv(
            "/Users/dimitrymindlin/UniProjects/Dialogue-XAI-APP/parsing/llm_intent_recognition/processed_xai_dataset_semicolon.csv",
            delimiter=";")
        result_df = intent_pd.copy()
        result_df['answer'] = None

        # Get all feature IDs and names for mapping
        feature_names = self.get_feature_names()
        feature_name_to_id = {item['feature_name']: item['id'] for item in feature_names}

        # Make sure we have a valid instance
        if self.current_instance is None:
            self.get_next_instance("train", 0)

        # Iterate through each row in the intent DataFrame
        for index, row in intent_pd.iterrows():
            xai_method = row['xai method']
            question = row['question']
            feature = row['feature'] if pd.notna(row['feature']) else None

            # Get question ID from the xai method
            question_id = xai_method

            if not question_id:
                result_df.at[index, 'answer'] = f"ERROR: Unknown XAI method '{xai_method}'"
                continue

            # Get feature ID if a feature is specified
            feature_id = None
            if feature is not None:
                if feature in feature_name_to_id:
                    feature_id = feature_name_to_id[feature]
                else:
                    # Try to find the closest match
                    import difflib
                    feature_names_list = list(feature_name_to_id.keys())
                    closest_matches = difflib.get_close_matches(feature, feature_names_list, n=1, cutoff=0.6)
                    if closest_matches:
                        feature_id = feature_name_to_id[closest_matches[0]]
                    else:
                        result_df.at[index, 'answer'] = f"ERROR: Unknown feature '{feature}'"
                        continue

            try:
                # Get answer for this combination
                answer, _, _, _ = self.update_state_new(question_id, feature_id)

                # Store answer in the result DataFrame
                result_df.at[index, 'answer'] = answer
                print(f"Processed: {xai_method} with feature {feature}")
            except Exception as e:
                # Record error in the DataFrame
                result_df.at[index, 'answer'] = f"ERROR: {str(e)}"
                print(f"Error processing {xai_method} with feature {feature}: {str(e)}")

        # Save the results to a CSV file
        result_df.to_csv(output_path, index=False, sep=";")
        print(f"All questions and answers saved to {output_path}")
        return result_df

    def trigger_background_xai_computation(self, phase_transition_from: str = "test"):
        """
        Trigger background computation of XAI reports for upcoming train instances.
        Call this when transitioning from test to train phase.
        
        Args:
            phase_transition_from: The phase we're transitioning from ("test", "intro-test", etc.)
        """
        if not self.use_llm_agent:
            return

        try:
            print(f"Triggering background XAI computation after {phase_transition_from} phase...")
            experiment_helper = self.conversation.get_var('experiment_helper').contents
            train_instances = experiment_helper.instances.get("train", [])

            if train_instances:
                # Get instance IDs for the first few train instances to precompute
                instance_ids_to_precompute = []
                max_precompute = min(5, len(train_instances))  # Precompute first 5 instances

                for i in range(max_precompute):
                    if i < len(train_instances):
                        instance_ids_to_precompute.append(train_instances[i].instance_id)

                # Submit for background computation
                self.xai_cache_manager.precompute_instances_background(
                    self.conversation,
                    instance_ids_to_precompute,
                    self.get_feature_display_name_dict()
                )
                print(f"Submitted {len(instance_ids_to_precompute)} instances for background computation")
            else:
                print("No train instances found for background computation")

        except Exception as e:
            print(f"Warning: Failed to trigger background XAI computation: {e}")

    def get_xai_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the XAI cache."""
        if hasattr(self, 'xai_cache_manager'):
            return self.xai_cache_manager.get_cache_stats()
        return {"error": "XAI cache manager not initialized"}

    def clear_xai_cache(self):
        """Clear the XAI cache."""
        if hasattr(self, 'xai_cache_manager'):
            self.xai_cache_manager.clear_cache()
            print("XAI cache cleared")
        else:
            print("XAI cache manager not initialized")
