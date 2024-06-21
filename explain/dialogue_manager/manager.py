from explain.dialogue_manager.dialogue_policy import DialoguePolicy


class DialogueManager:
    def __init__(self, intent_recognition, template_manager, active=True):
        # If the dialogue manager is active and suggests explanations or passive and just answers questions
        self.active_mode = active
        self.intent_recognition_model = intent_recognition
        self.template_manager = template_manager
        if self.active_mode:
            self.dialogue_policy = DialoguePolicy()
        else:
            self.dialogue_policy = None
        self.user_questions = 0
        self.interacted_explanations = set()
        self.most_important_attribute = None
        self.most_important_attribute_id = None
        self.current_attribute = None
        self.current_attribute_id = None

    def save_most_important_attribute(self, most_important_attribute):
        display_name = self.template_manager.get_feature_display_name_by_name(most_important_attribute)
        self.most_important_attribute = display_name
        self.most_important_attribute_id = most_important_attribute

    def save_current_attribute(self, current_attribute):
        display_name = self.template_manager.get_feature_display_name_by_name(current_attribute)
        self.current_attribute = display_name
        self.current_attribute_id = current_attribute

    def update_state(self, user_input, question_id=None, feature_id=None):
        """
        Update the state of the dialogue manager based on the user input. If the question_id is not None, the user
        clicked on a question and the dialogue manager can update the state machine directly. If the question_id is None,
        the user input needs NLU to determine the intent and the feature_id, then the state machine is updated.
        :param user_input: The user input
        :param question_id: The id of the question the user clicked on
        :param feature_id: The id of the feature the user clicked on
        :return: The id of the suggested explanation, the id of the feature the user clicked on, and the suggested followups
        """
        self.user_questions += 1
        if question_id is not None:
            # Direct mapping, update state machine
            self.interacted_explanations.add(question_id)
            if self.active_mode:
                self.dialogue_policy.model.trigger(question_id)
            return question_id, feature_id

        # If question_id is None, the user input needs NLU
        intent_classification = None
        method_name = None
        feature_name = None

        # Get user Intent
        if self.active_mode:
            intent_classification, method_name, feature_name = self.intent_recognition_model.interpret_user_answer(
                self.get_suggested_explanations(),
                user_input)

        if not self.active_mode or intent_classification == "other":
            method_name, feature_name = self.intent_recognition_model.predict_explanation_method(user_input)

        # Update the state machine
        if self.active_mode:
            self.dialogue_policy.model.trigger(method_name)
        return method_name, feature_name

    def replace_most_important_attribute(self, suggested_followups):
        # Replace "most important attribute" with the actual attribute name
        if self.most_important_attribute is not None:
            for followup in suggested_followups:
                # Feature specific questions have this placeholder
                if "most important attribute" in followup['question']:
                    followup['question'] = followup['question'].replace("most important attribute",
                                                                        self.most_important_attribute)
                    followup['feature'] = self.most_important_attribute_id
                if "current attribute" in followup['question']:
                    followup['question'] = followup['question'].replace("current attribute",
                                                                        self.current_attribute)
                    followup['feature'] = self.current_attribute_id
        return suggested_followups

    def get_suggested_explanations(self):
        suggested_followups = self.dialogue_policy.get_suggested_followups()
        suggested_followups = self.replace_most_important_attribute(suggested_followups)
        return suggested_followups

    def reset_state(self):
        """
        Reset the state of the dialogue manager for new interactions
        """
        self.dialogue_policy.reset_state()
        self.user_questions = 0
        self.interacted_explanations = set()
        self.most_important_attribute = None

    def print_transitions(self):
        self.dialogue_policy.to_mermaid()

    def get_proceeding_okay(self):
        """
        Checks if the user asked more than 2 questions. If yes, the user is okay with the explanation.
        If not, check which questions the user did not ask yet and return them.
        :return: Tuple of boolean and list of questions the user did not ask yet
        """
        if self.user_questions > 2:
            return True, None, ""
        else:
            if self.active_mode:
                not_asked_yet = self.dialogue_policy.get_not_asked_questions()
                not_asked_yet = self.replace_most_important_attribute(not_asked_yet)
                return False, not_asked_yet[:3], "Already want to proceed? Maybe the following could be interesting..."
            else:
                return False, None, "Already want to proceed? You could ask about the most or least important attribute," \
                                    "the influences of features or about which changes would lead to a different prediction..."
