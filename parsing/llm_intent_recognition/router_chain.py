from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

from prompt_toolkit import HTML, prompt

from parsing.llm_intent_recognition.prompts.explanations_prompt import get_xai_template_with_descriptions, ROUTING_TASK_PROMPT


class Config():
    LLM_MODEL = "llama3"
    llm = ChatOllama(model=LLM_MODEL, temperature=0)


cfg = Config()


class PromptFactory:
    # Define prompt names and descriptions
    prompt_names = [
        "xai_method", "whyExplanation", "notXaiMethod", "followUp", "greeting"
    ]

    prompt_descriptions = [
        "Prompt to select the most appropriate explainability method for the user's question about the model prediction. Useful when the user seeks explanations that can be answered by XAI methods like which conditions do not change prediction, feature importances, most important, least important features, counterfactuals, alternative scenarios, how to get different prediction, feature statistics: most common, min, max values for a feature.",
        "Prompt to address general 'why this prediction' questions, when user does only ask 'why'. Will list possible specific questions as an answer as solely 'why' cannot be answered directly.",
        "Prompt to respond to user questions unrelated to model predictions or feature specifics, which cannot be addressed using standard XAI methods.",
        "Prompt to handle follow-up questions that reference previous interactions, especially when the user shifts focus to a different feature or aspect. Useful for maintaining context continuity.",
        "Prompt to manage greetings and general inquiries not directly tied to the model predictions or specific features, ensuring a friendly and informative interaction."
    ]

    # Define the templates for each prompt
    # Define the templates for each prompt
    xai_method_template = get_xai_template_with_descriptions()

    why_template = """Turn this into a json and ONLY respond with the json output: "method_name": "whyExplanations", "feature": "null", """

    greeting_template = """Turn this into a json and ONLY respond with the json output: "method_name": "greeting", "feature": "null", """

    notXaiMethod_template = """Turn this into a json and ONLY respond with the json output: "method_name": "notXaiMethod", "feature": "null", """

    followUp_template = """Turn this into a json and ONLY respond with the json output: "method_name": "followUp", "feature": "null", """
    # Collect all templates in a list
    prompt_templates = [
        xai_method_template, why_template, notXaiMethod_template, followUp_template, greeting_template
    ]

    # Create prompt_infos dict with "name" and "description" and "prompt_template" keys
    prompt_infos = [
        {"name": name, "description": description, "prompt_template": template}
        for name, description, template in zip(prompt_names, prompt_descriptions, prompt_templates)
    ]


def generate_destination_chains():
    """
    Creates a list of LLM chains with different prompt templates.
    """
    prompt_factory = PromptFactory()
    destination_chains = {}
    for p_info in prompt_factory.prompt_infos:
        name = p_info['name']
        prompt_template = p_info['prompt_template']

        if name == "xai_method":
            chain_prompt_template = PromptTemplate.from_template(
                template=prompt_template, partial_variables={"format_instructions": format_instructions})
            chain = LLMChain(
                llm=cfg.llm,
                prompt=chain_prompt_template,
                output_parser=output_parser
            )
        else:
            chain = LLMChain(
                llm=cfg.llm,
                prompt=PromptTemplate(template=prompt_template, input_variables=['input']))
        destination_chains[name] = chain
    default_chain = ConversationChain(llm=cfg.llm)
    return prompt_factory.prompt_infos, destination_chains, default_chain


def generate_router_chain(prompt_infos, destination_chains, default_chain):
    """
    Generats the router chains from the prompt infos.
    :param prompt_infos The prompt informations generated above.
    """
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = '\n'.join(destinations)
    router_template = ROUTING_TASK_PROMPT.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=['input'],
        output_parser=RouterOutputParser()
    )
    router_chain = LLMRouterChain.from_llm(cfg.llm, router_prompt, verbose=False)
    return MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=False,
    )


if __name__ == "__main__":
    # Put here your API key or define it in your environment
    # os.environ["OPENAI_API_KEY"] = '<key>'

    prompt_infos, destination_chains, default_chain = generate_destination_chains()
    chain = generate_router_chain(prompt_infos, destination_chains, default_chain)
    while True:
        question = prompt(
            HTML("<b>Type <u>Your question</u></b>  ('q' to exit, 's' to save to html file): ")
        )
        if question == 'q':
            break
        result = chain.run(question)
        print(result)
        print()
