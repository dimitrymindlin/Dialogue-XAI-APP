"""
Utility functions for dual-mode LLM prediction supporting both structured output and string-based parsing.
This provides fallback mechanisms for LLMs that may not support structured output.
"""

import json
import logging
from typing import TypeVar, Type, Any, Dict, Union
from pydantic import BaseModel, ValidationError
from llama_index.core.llms.llm import LLM
from llama_index.core import PromptTemplate

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that may contain markdown code blocks or other formatting.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Cleaned JSON text string
    """
    json_text = text.strip()

    # Handle markdown code blocks
    if json_text.startswith("```json"):
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif json_text.startswith("```"):
        json_text = json_text.split("```")[1].split("```")[0].strip()

    return json_text


def create_pydantic_description_prompt(model_class: Type[T]) -> str:
    """
    Create a description of the pydantic model schema for use in string-based parsing prompts.
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        String description of the model schema
    """
    schema = model_class.model_json_schema()

    def format_property(name: str, prop_info: Dict[str, Any], required: bool = False) -> str:
        prop_type = prop_info.get('type', 'unknown')
        description = prop_info.get('description', 'No description available')
        required_text = " (REQUIRED)" if required else " (optional)"

        # Handle array types
        if prop_type == 'array':
            items = prop_info.get('items', {})
            if 'type' in items:
                prop_type = f"array of {items['type']}"
            elif '$ref' in items:
                prop_type = f"array of objects"

        # Handle object types with properties
        elif prop_type == 'object' and 'properties' in prop_info:
            prop_type = "object"

        return f"- {name}: {prop_type}{required_text} - {description}"

    # Get required fields
    required_fields = set(schema.get('required', []))

    # Format all properties
    properties = schema.get('properties', {})
    field_descriptions = []

    for field_name, field_info in properties.items():
        is_required = field_name in required_fields
        field_descriptions.append(format_property(field_name, field_info, is_required))

    description = f"""
Please respond with a JSON object that follows this exact schema for {model_class.__name__}:

{chr(10).join(field_descriptions)}

Make sure your response is valid JSON that can be parsed directly.
"""

    return description


async def astructured_predict_with_fallback(
        llm: LLM,
        model_class: Type[T],
        prompt_template: PromptTemplate,
        use_structured_output: bool = True,
        max_retries: int = 2,
        **prompt_kwargs
) -> T:
    """
    Attempt structured prediction with automatic fallback to string-based parsing.
    
    Args:
        llm: LLM instance
        model_class: Pydantic model class to predict
        prompt_template: Prompt template to use
        use_structured_output: If True, try structured output first; if False, use string parsing directly
        max_retries: Maximum number of retry attempts for string parsing
        **prompt_kwargs: Additional keyword arguments for prompt formatting
        
    Returns:
        Instance of model_class
        
    Raises:
        Exception: If both structured and string parsing fail
    """
    
    # Check if we should use structured output or go directly to string parsing
    if use_structured_output:
        # First attempt: Try structured prediction
        try:
            logger.info(f"Attempting structured prediction for {model_class.__name__}")
            result = await llm.astructured_predict(model_class, prompt_template, **prompt_kwargs)
            logger.info(f"Structured prediction successful for {model_class.__name__}")
            return result
            
        except Exception as structured_error:
            logger.warning(f"Structured prediction failed for {model_class.__name__}: {structured_error}")
            logger.info("Falling back to string-based parsing")
    else:
        logger.info(f"Using string-based parsing directly for {model_class.__name__} (structured_output=False)")
        structured_error = None  # No structured error since we didn't try it

        # Fallback: String-based parsing with model description
        try:
            # Create enhanced prompt with model description
            original_template = prompt_template.template
            model_description = create_pydantic_description_prompt(model_class)

            enhanced_template = f"{original_template}\n\n{model_description}"
            enhanced_prompt = PromptTemplate(enhanced_template)

            for attempt in range(max_retries):
                try:
                    logger.info(f"String parsing attempt {attempt + 1}/{max_retries} for {model_class.__name__}")

                    # Get raw text response
                    raw_response = await llm.apredict(enhanced_prompt, **prompt_kwargs)
                    logger.debug(f"Raw LLM response: {raw_response}")

                    # Extract and parse JSON
                    json_text = extract_json_from_text(raw_response)
                    parsed_data = json.loads(json_text)

                    # Validate with pydantic model
                    result = model_class(**parsed_data)
                    logger.info(f"String parsing successful for {model_class.__name__} on attempt {attempt + 1}")
                    return result

                except (json.JSONDecodeError, ValidationError, KeyError) as parse_error:
                    logger.warning(
                        f"String parsing attempt {attempt + 1} failed for {model_class.__name__}: {parse_error}")
                    if attempt == max_retries - 1:
                        raise parse_error
                    continue

        except Exception as fallback_error:
            error_msg = f"Failed to parse {model_class.__name__} using "
            if use_structured_output:
                error_msg += f"both structured output and string parsing. Structured error: {structured_error}. Fallback error: {fallback_error}"
            else:
                error_msg += f"string parsing. Error: {fallback_error}"

            logger.error(error_msg)
            raise Exception(error_msg)


def structured_predict_with_fallback(
        llm: LLM,
        model_class: Type[T],
        prompt_template: PromptTemplate,
        use_structured_output: bool = True,
        max_retries: int = 2,
        **prompt_kwargs
) -> T:
    """
    Synchronous version of structured prediction with fallback.
    
    Args:
        llm: LLM instance
        model_class: Pydantic model class to predict
        prompt_template: Prompt template to use
        use_structured_output: If True, try structured output first; if False, use string parsing directly
        max_retries: Maximum number of retry attempts for string parsing
        **prompt_kwargs: Additional keyword arguments for prompt formatting
        
    Returns:
        Instance of model_class
        
    Raises:
        Exception: If both structured and string parsing fail
    """

    # Check if we should use structured output or go directly to string parsing
    if use_structured_output:
        # First attempt: Try structured prediction
        try:
            logger.info(f"Attempting structured prediction for {model_class.__name__}")
            result = llm.structured_predict(model_class, prompt_template, **prompt_kwargs)
            logger.info(f"Structured prediction successful for {model_class.__name__}")
            return result

        except Exception as structured_error:
            logger.warning(f"Structured prediction failed for {model_class.__name__}: {structured_error}")
            logger.info("Falling back to string-based parsing")
    else:
        logger.info(f"Using string-based parsing directly for {model_class.__name__} (structured_output=False)")
        structured_error = None  # No structured error since we didn't try it

        # Fallback: String-based parsing with model description
        try:
            # Create enhanced prompt with model description
            original_template = prompt_template.template
            model_description = create_pydantic_description_prompt(model_class)

            enhanced_template = f"{original_template}\n\n{model_description}"
            enhanced_prompt = PromptTemplate(enhanced_template)

            for attempt in range(max_retries):
                try:
                    logger.info(f"String parsing attempt {attempt + 1}/{max_retries} for {model_class.__name__}")

                    # Get raw text response
                    raw_response = llm.predict(enhanced_prompt, **prompt_kwargs)
                    logger.debug(f"Raw LLM response: {raw_response}")

                    # Extract and parse JSON
                    json_text = extract_json_from_text(raw_response)
                    parsed_data = json.loads(json_text)

                    # Validate with pydantic model
                    result = model_class(**parsed_data)
                    logger.info(f"String parsing successful for {model_class.__name__} on attempt {attempt + 1}")
                    return result

                except (json.JSONDecodeError, ValidationError, KeyError) as parse_error:
                    logger.warning(
                        f"String parsing attempt {attempt + 1} failed for {model_class.__name__}: {parse_error}")
                    if attempt == max_retries - 1:
                        raise parse_error
                    continue

        except Exception as fallback_error:
            error_msg = f"Failed to parse {model_class.__name__} using "
            if use_structured_output:
                error_msg += f"both structured output and string parsing. Structured error: {structured_error}. Fallback error: {fallback_error}"
            else:
                error_msg += f"string parsing. Error: {fallback_error}"

            logger.error(error_msg)
            raise Exception(error_msg)
