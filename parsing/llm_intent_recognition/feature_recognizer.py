import spacy
from spacy.matcher import PhraseMatcher


class FeatureRecognizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')
        self.features = {
            'Age': ['age', 'years old', 'older', 'younger'],
            'Education Level': ['education level', 'educated', 'schooling'],
            'Marital Status': ['marital status', 'married', 'single', 'divorced'],
            'Occupation': ['occupation', 'job', 'profession', 'work'],
            'Weekly Working Hours': ['weekly working hours', 'hours worked', 'hours', 'working hours'],
            'Work Life Balance': ['work life balance', 'life balance', 'work and life balance'],
            'Investment Outcome': ['investment outcome', 'investment returns', 'financial performance']
        }
        for feature, patterns in self.features.items():
            phrases = [self.nlp.make_doc(text) for text in patterns]
            self.matcher.add(feature, phrases)

    def find_feature(self, user_input):
        doc = self.nlp(user_input.lower().strip())
        matches = self.matcher(doc)
        best_match = None
        max_length = 0
        for match_id, start, end in matches:
            length_of_match = end - start
            if length_of_match > max_length:
                max_length = length_of_match
                best_match = self.nlp.vocab.strings[match_id]
        return best_match
