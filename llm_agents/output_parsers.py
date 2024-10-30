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
