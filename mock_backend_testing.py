#!/usr/bin/env python3
"""
Mock backend testing for Dialogue-XAI-APP.
This script allows testing the performance of the backend without using the frontend.
It initializes the app, gets a training datapoint, and processes a list of user questions.
"""
import datetime
import json
import time
import argparse
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_BACKEND_URL = "http://localhost:4555"
DEFAULT_USER_ID = f"TEST_MOCK_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


class BackendMocker:
    """Class to interact with the Flask backend for performance testing."""

    def __init__(self, base_url: str, user_id: str):
        """Initialize the backend mocker.
        
        Args:
            base_url: Base URL of the Flask backend
            user_id: User ID to use for testing
        """
        self.base_url = base_url
        self.user_id = user_id
        self.session = requests.Session()
        # Timer for performance measurements
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        """Start the performance timer."""
        self.start_time = time.time()

    def end_timer(self) -> float:
        """End the timer and return the elapsed time."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        return elapsed

    def initialize_backend(self, study_group: str = "interactive", ml_knowledge: str = "low") -> Dict[str, Any]:
        """Initialize the backend by calling the init endpoint.
        
        Args:
            study_group: Study group to use (default: "interactive")
            ml_knowledge: ML knowledge level (default: "low")
            
        Returns:
            Dict with initialization data
        """
        logger.info(f"Initializing backend for user {self.user_id}")
        self.start_timer()
        response = self.session.get(
            f"{self.base_url}/init",
            params={"user_id": self.user_id, "study_group": study_group, "ml_knowledge": ml_knowledge}
        )
        elapsed = self.end_timer()

        if response.status_code != 200:
            logger.error(f"Failed to initialize backend: {response.status_code} {response.text}")
            raise Exception(f"Failed to initialize backend: {response.status_code}")

        logger.info(f"Backend initialized in {elapsed:.2f} seconds")
        return response.json()

    def get_train_datapoint(self, datapoint_count: int = 1) -> Dict[str, Any]:
        """Get a training datapoint.
        
        Args:
            datapoint_count: Datapoint count (1-indexed)
            
        Returns:
            Dict with datapoint data
        """
        logger.info(f"Getting training datapoint {datapoint_count}")
        self.start_timer()
        response = self.session.get(
            f"{self.base_url}/get_train_datapoint",
            params={"user_id": self.user_id, "datapoint_count": datapoint_count}
        )
        elapsed = self.end_timer()

        if response.status_code != 200:
            logger.error(f"Failed to get training datapoint: {response.status_code} {response.text}")
            raise Exception(f"Failed to get training datapoint: {response.status_code}")

        logger.info(f"Got training datapoint in {elapsed:.2f} seconds")
        return response.json()

    def set_user_prediction(self, prediction: str, experiment_phase: str = "teaching", datapoint_count: int = 1) -> \
    Dict[str, Any]:
        """Set the user prediction for the current datapoint.
        
        Args:
            prediction: User prediction (e.g. ">50K" or "<=50K")
            experiment_phase: Experiment phase ("teaching", "test", etc.)
            datapoint_count: Datapoint count (1-indexed)
            
        Returns:
            Dict with response data including initial message
        """
        logger.info(f"Setting user prediction: {prediction}")
        self.start_timer()
        response = self.session.post(
            f"{self.base_url}/set_user_prediction",
            json={
                "user_id": self.user_id,
                "experiment_phase": experiment_phase,
                "datapoint_count": datapoint_count,
                "user_prediction": prediction
            }
        )
        elapsed = self.end_timer()

        if response.status_code != 200:
            logger.error(f"Failed to set user prediction: {response.status_code} {response.text}")
            raise Exception(f"Failed to set user prediction: {response.status_code}")

        logger.info(f"Set user prediction in {elapsed:.2f} seconds")
        return response.json()

    async def ask_question_async(self, question: str) -> Dict[str, Any]:
        """Ask a natural language question to the backend asynchronously.
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with response data
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.base_url}/get_response_nl",
                    params={"user_id": self.user_id},
                    json={"message": question, "soundwave": False}
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Failed to ask question: {response.status} {text}")
                    raise Exception(f"Failed to ask question: {response.status}")

                data = await response.json()
                return data

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a natural language question to the backend.
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with response data
        """
        logger.info(f"Asking question: {question}")
        self.start_timer()
        response = self.session.post(
            f"{self.base_url}/get_response_nl",
            params={"user_id": self.user_id},
            json={"message": question, "soundwave": False}
        )
        elapsed = self.end_timer()

        if response.status_code != 200:
            logger.error(f"Failed to ask question: {response.status_code} {response.text}")
            raise Exception(f"Failed to ask question: {response.status_code}")

        logger.info(f"Got response in {elapsed:.2f} seconds")
        return response.json()

    async def ask_questions_in_sequence(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Ask a sequence of questions, waiting for each response before asking the next.
        
        Args:
            questions: List of natural language questions
            
        Returns:
            List of response data dictionaries
        """
        responses = []
        total_time_start = time.time()

        for i, question in enumerate(questions):
            logger.info(f"Asking question {i + 1}/{len(questions)}: {question}")
            start = time.time()
            response = await self.ask_question_async(question)
            elapsed = time.time() - start
            logger.info(f"Got response in {elapsed:.2f} seconds")
            responses.append(response)

        total_time = time.time() - total_time_start
        logger.info(
            f"Asked {len(questions)} questions in {total_time:.2f} seconds (avg: {total_time / len(questions):.2f}s)")
        return responses

    def finish(self):
        """Finish the experiment by calling the finish endpoint."""
        logger.info(f"Finishing experiment for user {self.user_id}")
        response = self.session.delete(
            f"{self.base_url}/finish",
            params={"user_id": self.user_id}
        )

        if response.status_code != 200:
            logger.error(f"Failed to finish experiment: {response.status_code} {response.text}")
            raise Exception(f"Failed to finish experiment: {response.status_code}")

        logger.info("Experiment finished")


async def main():
    """Main function to run the backend mocker."""
    parser = argparse.ArgumentParser(description="Test backend performance")
    parser.add_argument("--url", default=DEFAULT_BACKEND_URL, help="Backend URL")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="User ID for testing")
    parser.add_argument("--study-group", default="interactive", choices=["interactive", "chat", "static"],
                        help="Study group")
    parser.add_argument("--ml-knowledge", default="low", choices=["low", "medium", "high"], help="ML knowledge level")
    parser.add_argument("--prediction", default=">50K", help="User prediction to set")
    parser.add_argument("--questions-file", help="JSON file with questions to ask")
    args = parser.parse_args()

    # Initialize backend mocker
    mocker = BackendMocker(args.url, args.user_id)

    try:
        # Initialize backend
        init_data = mocker.initialize_backend(args.study_group, args.ml_knowledge)
        logger.info(f"Got {len(init_data['feature_names'])} features, {len(init_data['questions'])} questions")

        # Get training datapoint
        datapoint = mocker.get_train_datapoint(1)
        logger.info(f"Got datapoint: {json.dumps(datapoint, indent=2)[:100]}...")

        # Set user prediction
        prediction_response = mocker.set_user_prediction(args.prediction)
        if "initial_message" in prediction_response:
            logger.info(f"Initial message: {prediction_response['initial_message']['text'][:100]}...")

        # Load questions from file or use default questions
        if args.questions_file:
            with open(args.questions_file, 'r') as f:
                questions = json.load(f)
        else:
            questions = [
                "Why did the model predict this way?",
            ]
            # "Is Work Life balance important at all?"

        # Ask questions
        responses = await mocker.ask_questions_in_sequence(questions)

        # Print summary of responses
        logger.info(f"Asked {len(questions)} questions, got {len(responses)} responses")
        for i, (question, response) in enumerate(zip(questions, responses)):
            text = response.get('text', '')[:100] + '...' if len(response.get('text', '')) > 100 else response.get(
                'text', '')
            logger.info(f"Q{i + 1}: '{question}' -> '{text}'")

        # Finish experiment
        mocker.finish()

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Try to clean up
        try:
            mocker.finish()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())
