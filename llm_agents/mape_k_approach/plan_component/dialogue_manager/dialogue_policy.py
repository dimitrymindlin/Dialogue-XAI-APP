from transitions import Machine


class Model:
    pass


class DialoguePolicy:
    # Define the states based on explanations and steps
    states = [
        'initial',
        'shownCounterfactualsConcept',
        'shownImpactMultipleFeatures',
        'shownImpactSingleFeature',
        'shownFeatureImportancesConcept',
        'shownFeatureInfluencesPlot',
        'shownFeaturesInFavourOfOver50k',
        'shownFeaturesInFavourOfUnder50k',
        'shownWhyThisFeatureImportant',
        'shownAnchorExplanationConcept',
        'shownAnchor',
        'shownFeatureStatisticsConcept',
        'shownFeatureStatistics',
        'shownTextualPartialDependenceConcept',
        'shownPDPDescription',
        'shownPossibleClarificationsConcept',
        'shownClarificationsList',
        'shownModelPredictionConfidenceConcept',
        'shownConfidence',
        'shownCeterisParibusConcept',
        'shownPossibleClassFlips',
        'shownImpossibleClassFlips',
        'shownScaffoldingStrategyReformulating',
        'shownScaffoldingStrategyRepeating',
        'shownScaffoldingStrategyElicitingFeedback',
    ]

    # Define the method names as variables
    SHOW_COUNTERFACTUALS_CONCEPT = 'counterfactualsConcept'
    SHOW_IMPACT_MULTIPLE_FEATURES = 'impactMultipleFeatures'
    SHOW_IMPACT_SINGLE_FEATURE = 'impactSingleFeature'
    SHOW_FEATURE_IMPORTANCES_CONCEPT = 'featureImportancesConcept'
    SHOW_FEATURE_INFLUENCES_PLOT = 'featureInfluencesPlot'
    SHOW_FEATURES_IN_FAVOUR_OF_OVER_50K = 'featuresInFavourOfOver50k'
    SHOW_FEATURES_IN_FAVOUR_OF_UNDER_50K = 'featuresInFavourOfUnder50k'
    SHOW_WHY_THIS_FEATURE_IMPORTANT = 'whyThisFeatureImportant'
    SHOW_ANCHOR_EXPLANATION_CONCEPT = 'anchorExplanationConcept'
    SHOW_ANCHOR = 'anchor'
    SHOW_FEATURE_STATISTICS_CONCEPT = 'featureStatisticsConcept'
    SHOW_FEATURE_STATISTICS = 'featureStatistics'
    SHOW_TEXTUAL_PARTIAL_DEPENDENCE_CONCEPT = 'textualPartialDependenceConcept'
    SHOW_PDP_DESCRIPTION = 'pdpDescription'
    SHOW_POSSIBLE_CLARIFICATIONS_CONCEPT = 'possibleClarificationsConcept'
    SHOW_CLARIFICATIONS_LIST = 'clarificationsList'
    SHOW_MODEL_PREDICTION_CONFIDENCE_CONCEPT = 'modelPredictionConfidenceConcept'
    SHOW_CONFIDENCE = 'confidence'
    SHOW_CETERIS_PARIBUS_CONCEPT = 'ceterisParibusConcept'
    SHOW_POSSIBLE_CLASS_FLIPS = 'possibleClassFlips'
    SHOW_IMPOSSIBLE_CLASS_FLIPS = 'impossibleClassFlips'
    SHOW_SCAFFOLDING_STRATEGY_REFORMULATING = 'scaffoldingStrategyReformulating'
    SHOW_SCAFFOLDING_STRATEGY_REPEATING = 'scaffoldingStrategyRepeating'
    SHOW_SCAFFOLDING_STRATEGY_ELICITING_FEEDBACK = 'scaffoldingStrategyElicitingFeedback'

    # Define the transitions using the method name variables
    transitions = [
        {'trigger': SHOW_COUNTERFACTUALS_CONCEPT, 'source': '*', 'dest': 'shownCounterfactualsConcept'},
        {'trigger': SHOW_IMPACT_MULTIPLE_FEATURES, 'source': '*', 'dest': 'shownImpactMultipleFeatures'},
        {'trigger': SHOW_IMPACT_SINGLE_FEATURE, 'source': '*', 'dest': 'shownImpactSingleFeature'},
        {'trigger': SHOW_FEATURE_IMPORTANCES_CONCEPT, 'source': '*', 'dest': 'shownFeatureImportancesConcept'},
        {'trigger': SHOW_FEATURE_INFLUENCES_PLOT, 'source': '*', 'dest': 'shownFeatureInfluencesPlot'},
        {'trigger': SHOW_FEATURES_IN_FAVOUR_OF_OVER_50K, 'source': '*', 'dest': 'shownFeaturesInFavourOfOver50k'},
        {'trigger': SHOW_FEATURES_IN_FAVOUR_OF_UNDER_50K, 'source': '*', 'dest': 'shownFeaturesInFavourOfUnder50k'},
        {'trigger': SHOW_WHY_THIS_FEATURE_IMPORTANT, 'source': '*', 'dest': 'shownWhyThisFeatureImportant'},
        {'trigger': SHOW_ANCHOR_EXPLANATION_CONCEPT, 'source': '*', 'dest': 'shownAnchorExplanationConcept'},
        {'trigger': SHOW_ANCHOR, 'source': '*', 'dest': 'shownAnchor'},
        {'trigger': SHOW_FEATURE_STATISTICS_CONCEPT, 'source': '*', 'dest': 'shownFeatureStatisticsConcept'},
        {'trigger': SHOW_FEATURE_STATISTICS, 'source': '*', 'dest': 'shownFeatureStatistics'},
        {'trigger': SHOW_TEXTUAL_PARTIAL_DEPENDENCE_CONCEPT, 'source': '*',
         'dest': 'shownTextualPartialDependenceConcept'},
        {'trigger': SHOW_PDP_DESCRIPTION, 'source': '*', 'dest': 'shownPDPDescription'},
        {'trigger': SHOW_POSSIBLE_CLARIFICATIONS_CONCEPT, 'source': '*', 'dest': 'shownPossibleClarificationsConcept'},
        {'trigger': SHOW_CLARIFICATIONS_LIST, 'source': '*', 'dest': 'shownClarificationsList'},
        {'trigger': SHOW_MODEL_PREDICTION_CONFIDENCE_CONCEPT, 'source': '*',
         'dest': 'shownModelPredictionConfidenceConcept'},
        {'trigger': SHOW_CONFIDENCE, 'source': '*', 'dest': 'shownConfidence'},
        {'trigger': SHOW_CETERIS_PARIBUS_CONCEPT, 'source': '*', 'dest': 'shownCeterisParibusConcept'},
        {'trigger': SHOW_POSSIBLE_CLASS_FLIPS, 'source': '*', 'dest': 'shownPossibleClassFlips'},
        {'trigger': SHOW_IMPOSSIBLE_CLASS_FLIPS, 'source': '*', 'dest': 'shownImpossibleClassFlips'},
        {'trigger': SHOW_SCAFFOLDING_STRATEGY_REFORMULATING, 'source': '*',
         'dest': 'shownScaffoldingStrategyReformulating'},
        {'trigger': SHOW_SCAFFOLDING_STRATEGY_REPEATING, 'source': '*', 'dest': 'shownScaffoldingStrategyRepeating'},
        {'trigger': SHOW_SCAFFOLDING_STRATEGY_ELICITING_FEEDBACK, 'source': '*',
         'dest': 'shownScaffoldingStrategyElicitingFeedback'},
    ]

    # Define the questions for each transition
    questions = {
        SHOW_COUNTERFACTUALS_CONCEPT: "Would you like to understand the concept of counterfactual explanations?",
        SHOW_IMPACT_MULTIPLE_FEATURES: "Would you like to see the impact of multiple features?",
        SHOW_IMPACT_SINGLE_FEATURE: "Would you like to see the impact of a single feature?",
        SHOW_FEATURE_IMPORTANCES_CONCEPT: "Would you like to understand the concept of feature importances?",
        SHOW_FEATURE_INFLUENCES_PLOT: "Would you like to see the feature influences plot?",
        SHOW_FEATURES_IN_FAVOUR_OF_OVER_50K: "Would you like to see features in favor of over 50k?",
        SHOW_FEATURES_IN_FAVOUR_OF_UNDER_50K: "Would you like to see features in favor of under 50k?",
        SHOW_WHY_THIS_FEATURE_IMPORTANT: "Would you like to know why this feature is important?",
        SHOW_ANCHOR_EXPLANATION_CONCEPT: "Would you like to understand the concept of anchor explanations?",
        SHOW_ANCHOR: "Would you like to see the anchor explanation?",
        SHOW_FEATURE_STATISTICS_CONCEPT: "Would you like to understand the concept of feature statistics?",
        SHOW_FEATURE_STATISTICS: "Would you like to see the feature statistics?",
        SHOW_TEXTUAL_PARTIAL_DEPENDENCE_CONCEPT: "Would you like to understand the concept of textual partial dependence?",
        SHOW_PDP_DESCRIPTION: "Would you like to see the partial dependence plot description?",
        SHOW_POSSIBLE_CLARIFICATIONS_CONCEPT: "Would you like to understand the concept of possible clarifications?",
        SHOW_CLARIFICATIONS_LIST: "Would you like to see the list of possible clarifications?",
        SHOW_MODEL_PREDICTION_CONFIDENCE_CONCEPT: "Would you like to understand the concept of model prediction confidence?",
        SHOW_CONFIDENCE: "Would you like to see the model's prediction confidence?",
        SHOW_CETERIS_PARIBUS_CONCEPT: "Would you like to understand the concept of ceteris paribus?",
        SHOW_POSSIBLE_CLASS_FLIPS: "Would you like to see possible class flips?",
        SHOW_IMPOSSIBLE_CLASS_FLIPS: "Would you like to see impossible class flips?",
        SHOW_SCAFFOLDING_STRATEGY_REFORMULATING: "Would you like to see the reformulated explanation?",
        SHOW_SCAFFOLDING_STRATEGY_REPEATING: "Would you like to see the repeated explanation?",
        SHOW_SCAFFOLDING_STRATEGY_ELICITING_FEEDBACK: "Would you like to provide feedback on the explanation?",
    }

    followups = {
        'initial': [SHOW_COUNTERFACTUALS_CONCEPT, SHOW_FEATURE_IMPORTANCES_CONCEPT, SHOW_ANCHOR_EXPLANATION_CONCEPT],
        'shownCounterfactualsConcept': [SHOW_IMPACT_MULTIPLE_FEATURES, SHOW_IMPACT_SINGLE_FEATURE],
        'shownImpactMultipleFeatures': [SHOW_IMPACT_SINGLE_FEATURE],
        'shownImpactSingleFeature': [SHOW_FEATURE_IMPORTANCES_CONCEPT],
        'shownFeatureImportancesConcept': [SHOW_FEATURE_INFLUENCES_PLOT, SHOW_FEATURES_IN_FAVOUR_OF_OVER_50K,
                                           SHOW_FEATURES_IN_FAVOUR_OF_UNDER_50K],
        'shownFeatureInfluencesPlot': [SHOW_FEATURES_IN_FAVOUR_OF_OVER_50K, SHOW_FEATURES_IN_FAVOUR_OF_UNDER_50K],
        'shownFeaturesInFavourOfOver50k': [SHOW_FEATURES_IN_FAVOUR_OF_UNDER_50K],
        'shownFeaturesInFavourOfUnder50k': [SHOW_WHY_THIS_FEATURE_IMPORTANT],
        'shownWhyThisFeatureImportant': [SHOW_ANCHOR_EXPLANATION_CONCEPT],
        'shownAnchorExplanationConcept': [SHOW_ANCHOR],
        'shownAnchor': [SHOW_FEATURE_STATISTICS_CONCEPT],
        'shownFeatureStatisticsConcept': [SHOW_FEATURE_STATISTICS],
        'shownFeatureStatistics': [SHOW_TEXTUAL_PARTIAL_DEPENDENCE_CONCEPT],
        'shownTextualPartialDependenceConcept': [SHOW_PDP_DESCRIPTION],
        'shownPDPDescription': [SHOW_POSSIBLE_CLARIFICATIONS_CONCEPT],
        'shownPossibleClarificationsConcept': [SHOW_CLARIFICATIONS_LIST],
        'shownClarificationsList': [SHOW_MODEL_PREDICTION_CONFIDENCE_CONCEPT],
        'shownModelPredictionConfidenceConcept': [SHOW_CONFIDENCE],
        'shownConfidence': [SHOW_CETERIS_PARIBUS_CONCEPT],
        'shownCeterisParibusConcept': [SHOW_POSSIBLE_CLASS_FLIPS, SHOW_IMPOSSIBLE_CLASS_FLIPS],
        'shownPossibleClassFlips': [SHOW_IMPOSSIBLE_CLASS_FLIPS],
        'shownImpossibleClassFlips': [SHOW_SCAFFOLDING_STRATEGY_REFORMULATING],
        'shownScaffoldingStrategyReformulating': [SHOW_SCAFFOLDING_STRATEGY_REPEATING],
        'shownScaffoldingStrategyRepeating': [SHOW_SCAFFOLDING_STRATEGY_ELICITING_FEEDBACK],
        'shownScaffoldingStrategyElicitingFeedback': [],
    }

    def __init__(self):
        # Create the state machine
        self.model = Model()
        self.machine = Machine(model=self.model, states=DialoguePolicy.states, initial='initial')
        self.asked_questions = []

    def compute_dynamic_modifier(self, transition, user_explanations: dict, ml_knowledge: str):
        """Compute a dynamic modifier for the transition probability based on the user's explanations and ML knowledge."""
        modifier = 1.0
        dest = transition['dest']

        # If the explanation for the destination is already marked as UNDERSTOOD, reduce its weight
        if dest in user_explanations and user_explanations[dest] == 'UNDERSTOOD':
            modifier *= 0.5

        ml_level = ml_knowledge.lower()
        if ml_level in ['advanced', 'expert']:
            # For advanced users, lower weight on basic concept explanations
            if 'concept' in dest.lower():
                modifier *= 0.8
            else:
                modifier *= 1.2
        elif ml_level in ['beginner']:
            # For beginners, favor concept explanations
            if 'concept' in dest.lower():
                modifier *= 1.2
            else:
                modifier *= 0.8

        return modifier

    def trigger_transition(self, trigger, user_explanations: dict, ml_knowledge: str):
        import random
        current_state = self.model.state
        possible_transitions = [t for t in DialoguePolicy.transitions if
                                t['trigger'] == trigger and (t['source'] == '*' or t['source'] == current_state)]
        if not possible_transitions:
            print(f"No valid transition for trigger: {trigger}")
            return
        if len(possible_transitions) == 1:
            chosen = possible_transitions[0]
        else:
            weights = []
            for t in possible_transitions:
                base = t.get('probability', 1.0)
                modifier = self.compute_dynamic_modifier(t, user_explanations, ml_knowledge)
                weights.append(base * modifier)
            total = sum(weights)
            normalized = [w / total for w in weights]
            chosen = random.choices(possible_transitions, weights=normalized, k=1)[0]
        self.machine.set_state(chosen['dest'])
        self.ask_question(trigger=trigger)

    def ask_question(self, *args, **kwargs):
        trigger = kwargs.get('trigger')
        if trigger:
            self.asked_questions.append(trigger)
            print(DialoguePolicy.questions[trigger])

    def get_suggested_followups(self):
        current_state = self.model.state
        return [{'id': trigger, 'question': DialoguePolicy.questions[trigger], 'feature_id': None} for trigger in
                DialoguePolicy.followups.get(current_state, [])]

    def get_last_explanation(self):
        try:
            return self.asked_questions[-1]
        except IndexError:
            return None

    def get_not_asked_questions(self, num_questions=2):
        """
        Get the next num_questions that were not asked yet, avoiding follow ups from the current state.
        If there are not enough not asked questions, get the rest from the follow ups.
        return: List of dictionaries with id, question and feature keys
        """
        followups = DialoguePolicy.followups.get(self.model.state, [])
        not_asked_questions = [q for q in DialoguePolicy.questions if
                               q not in self.asked_questions and q not in followups]

        # Return required not asked questions, filling the rest with followups if needed
        questions = not_asked_questions[:num_questions]
        if len(questions) < num_questions:
            questions.extend(followups[:num_questions - len(not_asked_questions)])

        return [{'id': q, 'question': DialoguePolicy.questions[q], 'feature': None} for q in questions]

    def get_explanation_plan(self, num_steps: int, user_explanations: dict, ml_knowledge: str):
        """
        Compute a dynamic explanation plan (sequence of explanation steps) ahead of time.
        Starting from the current state, it simulates num_steps transitions using the dynamic
        probabilities computed from user_explanations and ml_knowledge.
        Returns a list of steps where each step is a dict with trigger, question, and destination state.
        """
        import random
        plan = []
        simulated_state = self.model.state
        for _ in range(num_steps):
            possible_transitions = [t for t in DialoguePolicy.transitions if
                                    t['trigger'] and (t['source'] == '*' or t['source'] == simulated_state)]
            if not possible_transitions:
                break
            if len(possible_transitions) == 1:
                chosen = possible_transitions[0]
            else:
                weights = []
                for t in possible_transitions:
                    base = t.get('probability', 1.0)
                    modifier = self.compute_dynamic_modifier(t, user_explanations, ml_knowledge)
                    weights.append(base * modifier)
                total = sum(weights)
                normalized = [w / total for w in weights]
                chosen = random.choices(possible_transitions, weights=normalized, k=1)[0]
            trigger = chosen['trigger']
            question = DialoguePolicy.questions.get(trigger, '')
            plan.append({'trigger': trigger, 'question': question, 'dest': chosen['dest']})
            simulated_state = chosen['dest']
        return plan

    def reset_state(self):
        """
        Reset the state of the dialogue policy for new interactions
        """
        self.machine.set_state('initial')

    def to_mermaid(self, include_trigger=False):
        mermaid_def = "stateDiagram-v2\n"
        for state in self.states:
            state = state.replace('shown', '')
            mermaid_def += f"    {state}\n"
        for state, followup_triggers in self.followups.items():
            for trigger in followup_triggers:
                dest = next(t['dest'] for t in self.transitions if t['trigger'] == trigger)
                state = state.replace('shown', '')
                dest = dest.replace('shown', '')

                if include_trigger:
                    mermaid_def += f"    {state} --> {dest} : {trigger}\n"
                else:
                    mermaid_def += f"    {state} --> {dest}\n"
        print(mermaid_def)
