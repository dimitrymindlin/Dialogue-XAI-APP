from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def get_analysis_output_parser():
    analysis_response_schemas = [
        ResponseSchema(name="analysis", description="Reasoning over the selection of xAI methods."),
        ResponseSchema(name="selected_methods", description="List of selected xAI methods."),
    ]
    analysis_output_parser = StructuredOutputParser.from_response_schemas(analysis_response_schemas)
    return analysis_output_parser


def get_response_output_parser():
    response_response_schemas = [
        ResponseSchema(name="response", description="Concise response to the user's question."),
    ]
    response_output_parser = StructuredOutputParser.from_response_schemas(response_response_schemas)
    return response_output_parser


def get_mape_k_monitor_output_parser():
    mape_k_monitor_response_schemas = [
        ResponseSchema(name="monitor_result",
                       description="classification of: understanding, misunderstanding or neutral"),
    ]
    mape_k_monitor_output_parser = StructuredOutputParser.from_response_schemas(mape_k_monitor_response_schemas)
    return mape_k_monitor_output_parser


def get_mape_k_analyze_output_parser():
    mape_k_analyze_response_schemas = [
        ResponseSchema(name="analysis",
                       description="Your reasoning over what parts the user understood or misunderstood. Analyzing the user's understanding."),
        ResponseSchema(name="user_understanding_notepad",
                       description="user_understanding_notepad dict with the keys: {understood,\n misunderstood,\n not explained yet,\n additional_comments}"),
    ]
    mape_k_analyze_output_parser = StructuredOutputParser.from_response_schemas(mape_k_analyze_response_schemas)
    return mape_k_analyze_output_parser


def get_mape_k_plan_output_parser():
    map_k_plan_response_schemas = [
        ResponseSchema(name="reasoning", description="Reasoning over which xAI methods to use next."),
        ResponseSchema(name="next_explanation", description="The next explanation to give to the user"),
    ]
    map_k_plan_output_parser = StructuredOutputParser.from_response_schemas(map_k_plan_response_schemas)
    return map_k_plan_output_parser


def get_mape_k_execute_output_parser():
    map_k_execute_response_schemas = [
        ResponseSchema(name="reasoning",
                       description="Your reasoning over how to answer the user question given his understanding and the planned action."),
        ResponseSchema(name="response", description="The final response to the user message"),
    ]
    map_k_execute_output_parser = StructuredOutputParser.from_response_schemas(map_k_execute_response_schemas)
    return map_k_execute_output_parser
