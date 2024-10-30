class MapeKExecute:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def execute(self, plan_response):
        action = plan_response['action']
        method_name = plan_response.get('method')
        feature_name = plan_response.get('feature')
        reasoning = plan_response.get('reasoning')

        if 'Provide a new explanation' in action:
            explanation = self.generate_explanation(method_name, feature_name)
            self.knowledge_base.last_explanation = explanation
            return explanation
        elif 'Clarify the previous explanation' in action:
            explanation = self.generate_clarification(self.knowledge_base.last_explanation)
            return explanation
        elif 'Ask a question' in action:
            question = self.generate_question(reasoning)
            return question
        else:
            return "I'm sorry, could you please clarify your question?"

    def generate_explanation(self, method_name, feature_name):
        # Replace with actual explanation logic
        explanation = f"Here's an explanation using {method_name} for the feature {feature_name}."
        return explanation

    def generate_clarification(self, last_explanation):
        clarification = f"Let me clarify: {last_explanation}"
        return clarification

    def generate_question(self, reasoning):
        question = f"I noticed you might be unsure about {reasoning}. Could you tell me more about what confuses you?"
        return question
