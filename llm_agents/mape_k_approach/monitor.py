from user_modelling.monitor_llm import MonitorLLM


class MapeKMonitor:
    def __init__(self, feature_names):
        self.llm = MonitorLLM(feature_names)
        self.chat_history = []

    def monitor_user_input(self, user_input):
        monitor_response = self.llm.monitor_user_input(user_input, self.chat_history)
        # Update chat history
        self.chat_history.append({
            'user_input': user_input,
            'assistant_response': monitor_response  # This is the monitoring output
        })
        return monitor_response
