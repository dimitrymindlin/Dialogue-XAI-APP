from user_modelling.analyze import MapeKAnalyze
from user_modelling.execute import MapeKExecute
from user_modelling.knowledge_base import MapeKKnowledgeBase
from user_modelling.monitor import MapeKMonitor
from user_modelling.plan import MapeKPlan


class MapeKSystem:
    def __init__(self, feature_names):
        self.knowledge_base = MapeKKnowledgeBase()
        self.monitor = MapeKMonitor(feature_names)
        self.analyze = MapeKAnalyze(self.knowledge_base)
        self.plan = MapeKPlan(self.knowledge_base)
        self.execute = MapeKExecute(self.knowledge_base)
        self.feature_names = feature_names

    def update_state(self, user_input, current_classification):
        explanation_suggestions = self.get_suggested_explanations()
        # Monitor User Question
        monitor_response = self.monitor.monitor_user_input(user_input)
        # Update Knowledge Base
        """self.knowledge_base.user_model.update_understanding(monitor_response["concepts_learned"],
                                                            monitor_response["misconceptions"])
            self.knowledge_base.dialogue_history.append(monitor_response)"""
        self.knowledge_base.user_model.update_from_monitor(monitor_response)
        # Analyze User Question
        self.analyze.analyze_user_input(user_input, self.knowledge_base)
        # Update Knowledge Base
        self.knowledge_base.update_from_analyze(self.analyze)
        # Plan Next Action
        plan_response = self.plan.plan_next_action(monitor_response)
        response = self.execute.execute(plan_response)
        return response

    def get_suggested_explanations(self):
        # Implement logic to get suggested explanations
        return [
            {'id': 'top3Features', 'question': 'Would you like to see the top 3 important attributes?',
             'feature': None},
            # Add more suggestions as needed
        ]
