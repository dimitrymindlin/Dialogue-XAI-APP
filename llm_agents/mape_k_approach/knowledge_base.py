from user_modelling.user_model import MapeKUserModel


class MapeKKnowledgeBase:
    def __init__(self):
        self.user_model = MapeKUserModel()
        self.last_explanation = ""
        self.last_method = None

    def update_from_analyze(self, analyze_response):
        self.last_method = analyze_response.get('method')
        self.last_feature = analyze_response.get('feature')

    def to_text(self):
        understanding = self.user_model.understood_concepts
        misconceptions = self.user_model.misconceptions
        xai_knowledge = self.user_model.xai_knowledge
        return f"Understanding: {understanding}, Misconceptions: {misconceptions}, XAI Knowledge: {xai_knowledge}"
