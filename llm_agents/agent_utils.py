"""
Shared utilities for XAI agents.

This module contains common functionality used across different agent implementations
(LlamaIndex and OpenAI), including logging setup, CSV handling, and timing decorators.
"""
import csv
import functools
import logging
import os
from datetime import datetime
from flask import current_app
from dotenv import load_dotenv
from typing import Dict, Any, Callable

# Load environment variables once during module import
load_dotenv()

# Configure environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION_ID")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_MINI_MODEL_NAME = os.getenv("OPENAI_MINI_MODEL_NAME")
OPENAI_REASONING_MODEL_NAME = os.getenv("OPENAI_REASONING_MODEL_NAME")

# Base directory: the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for CSV logs: use project-root “cache” folder
LOG_FOLDER = os.path.join(os.getcwd(), "cache")
os.makedirs(LOG_FOLDER, exist_ok=True)
# CSV headers for one-row-per-run logging: timestamp, experiment_id, and JSON list of items
CSV_HEADERS = ["timestamp", "experiment_id", "items"]


def configure_logger(name: str, log_file: str = "agent.log") -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.
    
    Args:
        name: The name of the logger
        log_file: The file to write logs to
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)


    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))

        # File handler (DISABLED)
        # file_handler = logging.FileHandler(log_file, mode="w")
        # file_handler.setLevel(logging.INFO)
        # file_handler.setFormatter(logging.Formatter(
        #     fmt="%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n"
        # ))

        logger.addHandler(console_handler)
        # logger.addHandler(file_handler)

    return logger


def timed(fn: Callable) -> Callable:
    """
    Decorator to time async function execution and log the result.
    
    Args:
        fn: The async function to time
        
    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(fn)
    async def wrapper(self, *args, **kwargs):
        start = datetime.now()
        result = await fn(self, *args, **kwargs)
        elapsed = datetime.now() - start
        
        # Try to use Flask logger, fallback to standard logging if not available
        try:
            if current_app:
                current_app.logger.info(f"{self.__class__.__name__} answered in {elapsed}")
            else:
                raise RuntimeError("Flask not available")
        except (RuntimeError, AttributeError):
            # Working outside of application context or Flask not available - use standard logger
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"{self.__class__.__name__} answered in {elapsed}")
        
        return result

    return wrapper


def generate_log_file_name(experiment_id: str) -> str:
    """
    Generate a fixed log file name for all LLM executions.
    
    Args:
        experiment_id: The unique experiment identifier (not used for filename, but kept for compatibility)
        
    Returns:
        The absolute path to the log file
    """
    return os.path.join(LOG_FOLDER, "llm_executions.csv")


def initialize_csv(log_file: str) -> None:
    """
    Initialize a CSV log file with headers if it doesn't exist.
    
    Args:
        log_file: The path to the CSV file to initialize
    """
    if not os.path.isfile(log_file):
        try:
            with open(log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(CSV_HEADERS)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to initialize CSV: {e}")


def append_new_log_row(row: Dict[str, Any], log_file: str) -> None:
    """
    Append a new row to the log CSV file.
    
    Args:
        row: A dictionary mapping column headers to values
        log_file: The path to the CSV file
    """
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([row.get(header, "") for header in CSV_HEADERS])


def update_last_log_row(row: Dict[str, Any], log_file: str) -> None:
    """
    Update the last row in the log CSV file.
    
    Args:
        row: A dictionary with updated values for specific columns
        log_file: The path to the CSV file
    """
    with open(log_file, 'r', newline='', encoding='utf-8') as file:
        lines = list(csv.reader(file, delimiter=';'))

    if len(lines) < 2:
        return  # Nothing to update

    # Update values in the last row
    lines[-1] = [str(row.get(h, lines[-1][i])) for i, h in enumerate(CSV_HEADERS)]

    with open(log_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(lines)


def log_prompt(prompt_log_file: str, component: str, prompt_str: str) -> None:
    """
    Append a prompt to the prompt log file.
    
    Args:
        prompt_log_file: Path to the prompt log file
        component: The component name (e.g., 'monitor', 'analyze')
        prompt_str: The raw prompt content
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"{timestamp} - {component.upper()} prompt:\n{prompt_str}\n\n"

    with open(prompt_log_file, 'a', encoding='utf-8') as f:
        f.write(entry)

    logging.getLogger(__name__).info(f"{component.capitalize()} prompt logged")


def get_definition_paths() -> Dict[str, str]:
    """
    Get the paths to shared definition files.
    
    Returns:
        A dictionary mapping definition names to file paths
    """
    return {
        "understanding_displays": os.path.join(
            BASE_DIR, "mape_k_approach", "monitor_component", "understanding_displays_definition.json"
        ),
        "modes_of_engagement": os.path.join(
            BASE_DIR, "mape_k_approach", "monitor_component", "icap_modes_definition.json"
        ),
        "explanation_questions": os.path.join(
            BASE_DIR, "mape_k_approach", "monitor_component", "explanation_questions_definition.json"
        ),
    }
