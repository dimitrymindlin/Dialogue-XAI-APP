class MapeKUserModel:
    def __init__(self):
        self.understood_concepts = set()
        self.misconceptions = set()
        self.learning_history = []

    def update_understanding(self, concepts_learned, misconceptions):
        self.understood_concepts.update(concepts_learned)
        self.misconceptions.update(misconceptions)
        self.learning_history.append({
            'concepts_learned': concepts_learned,
            'misconceptions': misconceptions
        })

    def update_from_monitor(self, monitor_response):
        thoughts = monitor_response.get('thoughts', '')
        xai_knowledge = monitor_response.get('xAIknowledge', '')
        concepts_learned = monitor_response.get('concepts_learned', [])
        misconceptions = monitor_response.get('misconceptions', [])

        self.xai_knowledge = xai_knowledge
        self.understood_concepts.update(concepts_learned)
        self.misconceptions.update(misconceptions)
        self.learning_history.append({
            'thoughts': thoughts,
            'concepts_learned': concepts_learned,
            'misconceptions': misconceptions
        })
