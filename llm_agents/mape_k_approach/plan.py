from user_modelling.plan_llm import PlanLLM


class MapeKPlan:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.llm = PlanLLM()

    def plan_next_action(self, analyze_response):
        user_model = self.knowledge_base.user_model
        understood_concepts = ', '.join(user_model.understood_concepts)
        misconceptions = ', '.join(user_model.misconceptions)
        xai_knowledge = user_model.xai_knowledge
        last_explanation = self.knowledge_base.last_explanation
        method = analyze_response.get('method')
        feature = analyze_response.get('feature')

        # Use PlanLLM to decide next action based on analysis and user model
        plan_response = self.llm.plan_next_action(
            method,
            feature,
            understood_concepts,
            misconceptions,
            xai_knowledge,
            last_explanation
        )
        return plan_response
