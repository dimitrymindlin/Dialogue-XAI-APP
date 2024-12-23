from enum import Enum


class ExplanationState(Enum):
    NOT_YET_EXPLAINED = "not_yet_explained"
    SHOWN = "shown"
    UNDERSTOOD = "understood"
    NOT_UNDERSTOOD = "not_understood"
    PARTIALLY_UNDERSTOOD = "partially_understood"
