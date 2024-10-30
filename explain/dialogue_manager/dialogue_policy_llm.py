class DialoguePolicyLLM:
    def __init__(self):
        # Initialize any required variables
        self.asked_questions = []
        self.last_method = None

    def plan_next_action(self, user_model, last_explanation, user_intent):
        """
        Use the LLM to decide the next action based on the user's understanding and misconceptions.
        """
        understood_concepts = ', '.join(user_model.understood_concepts)
        misconceptions = ', '.join(user_model.misconceptions)
        prompt = f"""
You are an AI assistant helping a user to understand a machine learning model.
The user has the following intent: "{user_intent}".
They currently understand: {understood_concepts}.
They may have misconceptions about: {misconceptions}.
The last explanation provided was: {last_explanation}.

Based on this information, decide whether to:
- Provide a new explanation (specify which method and feature).
- Clarify the previous explanation.
- Ask a question to gauge understanding.

Provide your decision and reasoning.
"""
        response = call_llm(prompt)
        # Parse the response to extract the decision
        decision = self.parse_llm_response(response)
        return decision

    def parse_llm_response(self, response):
        """
        Parse the LLM response to extract the decision.
        """
        # For simplicity, assume the response is structured
        # In practice, you might need to parse the text more carefully or use structured outputs
        lines = response.strip().split('\n')
        decision = lines[0].replace('Decision:', '').strip()
        reasoning = lines[1].replace('Reasoning:', '').strip()
        method_name = None
        feature_name = None

        if 'Provide a new explanation' in decision:
            # Extract method and feature if provided
            if 'Method:' in decision:
                method_name = decision.split('Method:')[1].split(',')[0].strip()
            if 'Feature:' in decision:
                feature_name = decision.split('Feature:')[1].strip()
        elif 'Clarify the previous explanation' in decision:
            method_name = self.last_method  # Use the last method for clarification

        return {
            'action': decision,
            'reasoning': reasoning,
            'method_name': method_name,
            'feature_name': feature_name
        }

    def update_last_method(self, method_name):
        self.last_method = method_name
