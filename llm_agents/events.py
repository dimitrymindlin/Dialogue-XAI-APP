from llama_index.core.workflow import Event


class MonitorDoneEvent(Event):
    pass


class AnalyzeDoneEvent(Event):
    pass


class PlanDoneEvent(Event):
    pass
