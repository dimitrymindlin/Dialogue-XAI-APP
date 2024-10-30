from user_modelling.analyze_llm import AnalyzeLLM


class MapeKAnalyze:
    def __init__(self, feature_names):
        self.llm = AnalyzeLLM(feature_names)
        self.chat_history = []

    def analyze_user_input(self, user_input, knowledge_base):
        analyze_response = self.llm.analyze_user_input(user_input, self.chat_history, knowledge_base.to_text())
        # Update chat history
        self.chat_history.append({
            'user_input': user_input,
            'assistant_response': analyze_response
        })
        return analyze_response
