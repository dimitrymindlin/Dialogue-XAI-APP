#!/usr/bin/env python3
"""
Mock backend testing for Dialogue-XAI-APP.
This script allows testing the performance of the backend without using the frontend.
It initializes the app, gets a training datapoint, and processes a list of user questions.
Now with STREAMING support for testing LLM agents directly!
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

    async def ask_question_stream(self, question: str) -> Dict[str, Any]:
        """Ask a natural language question to the backend with streaming response.
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with final response data
        """
        logger.info(f"Asking streaming question: {question}")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/get_response_nl_stream",  # Correct streaming endpoint
                params={"user_id": self.user_id},
                json={"message": question, "soundwave": False},
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Failed to ask streaming question: {response.status} {text}")
                    raise Exception(f"Failed to ask streaming question: {response.status}")

                full_response = ""
                final_data = None
                chunk_count = 0
                
                logger.info("Starting to receive streamed response...")
                print(f"\n[STREAM] Question: {question}")
                print("[STREAM] Response: ", end="", flush=True)
                
                # Read Server-Sent Events stream
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        try:
                            chunk_data = json.loads(data_str)
                            chunk_count += 1
                            
                            if chunk_data.get('type') == 'partial':
                                content = chunk_data.get('content', '')
                                print(content, end='', flush=True)
                                full_response += content
                                
                            elif chunk_data.get('type') == 'final':
                                final_content = chunk_data.get('content', '')
                                print(f"{final_content}")
                                final_data = chunk_data
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON chunk: {data_str[:100]}... Error: {e}")
                            continue
                            
                    elif line_str == "":
                        # Empty line, continue reading
                        continue
                        
                elapsed = time.time() - start_time
                logger.info(f"Streaming completed in {elapsed:.2f} seconds ({chunk_count} chunks)")
                
                # Return final data or construct response from accumulated content
                if final_data:
                    return final_data
                else:
                    return {
                        "text": full_response,
                        "reasoning": "Stream completed without final chunk",
                        "type": "stream_complete"
                    }

    async def test_direct_agent_stream(self, question: str) -> Dict[str, Any]:
        """Test streaming directly from MAPE-K agents without Flask backend.
        
        This is a NEW method that tests the streaming functionality directly 
        from the updated mape_k_mixins.py agents.
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with final response data
        """
        logger.info(f"Testing DIRECT AGENT STREAMING for question: {question}")
        start_time = time.time()
        
        try:
            # Import the streaming-enabled agents
            from llm_agents.mape_k_mixins import MapeK2BaseAgent, MapeKUnifiedBaseAgent
            from llama_index.llms.openai import OpenAI
            import os
            
            # Create agent (you can test different agent types)
            agent = MapeK2BaseAgent(
                llm=OpenAI(model="gpt-4o"),
                experiment_id="stream_test",
                feature_names="age,workclass,education,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country",
                domain_description="Adult income prediction dataset",
                user_ml_knowledge="low",
                timeout=120.0
            )
            
            print(f"\n[DIRECT STREAM] Question: {question}")
            print("[DIRECT STREAM] Response: ", end="", flush=True)
            
            full_response = ""
            final_data = None
            chunk_count = 0
            chunk_times = []  # Track timing for each chunk
            first_chunk_time = None
            
            # Use the new streaming method
            async for chunk in agent.answer_user_question_stream(question):
                chunk_count += 1
                current_time = time.time() - start_time
                
                print(f"\n🔍 DEBUG: Received chunk #{chunk_count}: type={chunk.get('type')}, content_length={len(chunk.get('content', ''))}")
                
                if chunk.get('type') == 'partial':
                    content = chunk.get('content', '')
                    print(f"🔍 DEBUG: Partial content (first 100 chars): '{content[:100]}...'")
                    print(content, end='', flush=True)
                    full_response += content
                    
                    # Record timing for this chunk
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        chunk_times.append(f"1st Chunk -> {first_chunk_time:.2f}s")
                        print(f" [⚡ 1st chunk in {first_chunk_time:.2f}s]", end="", flush=True)
                    else:
                        chunk_times.append(f"{self._ordinal(chunk_count)} chunk -> {current_time:.2f}s")
                        print(f" [📦 {self._ordinal(chunk_count)} chunk at {current_time:.2f}s]", end="", flush=True)
                    
                elif chunk.get('type') == 'final':
                    final_content = chunk.get('content', '')
                    print(f"\n🔍 DEBUG: Final content length: {len(final_content)}")
                    print(f"🔍 DEBUG: Final vs accumulated same? {final_content == full_response}")
                    
                    if final_content and final_content != full_response:
                        print(final_content)
                        full_response = final_content
                    else:
                        print()  # New line
                    final_data = chunk
                    break
                    
                elif chunk.get('type') == 'error':
                    error_msg = chunk.get('content', 'Unknown error')
                    print(f"\n[ERROR] {error_msg}")
                    return {"error": error_msg, "type": "error"}
            
            elapsed = time.time() - start_time
            
            # Create detailed timing summary
            timing_summary = " | ".join(chunk_times) if chunk_times else "No chunks received"
            
            logger.info(f"📊 Streaming Timeline: {timing_summary}")
            logger.info(f"⏱️  Total Time: {elapsed:.2f}s | Chunks: {chunk_count} | First Response: {first_chunk_time:.2f}s")
            
            return {
                "text": final_data.get('content', full_response) if final_data else full_response,
                "reasoning": final_data.get('reasoning', 'No reasoning available') if final_data else 'No reasoning available',
                "type": "direct_stream_complete",
                "elapsed_time": elapsed,
                "chunk_count": chunk_count,
                "first_chunk_time": first_chunk_time,
                "chunk_times": chunk_times,
                "timing_summary": timing_summary
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Direct agent streaming failed after {elapsed:.2f} seconds: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "type": "direct_stream_error",
                "elapsed_time": elapsed
            }

    def _ordinal(self, n):
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

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

    async def ask_questions_stream_sequence(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Ask a sequence of questions with streaming, waiting for each response before asking the next.
        
        Args:
            questions: List of natural language questions
            
        Returns:
            List of response data dictionaries
        """
        responses = []
        total_time_start = time.time()

        for i, question in enumerate(questions):
            logger.info(f"Asking streaming question {i + 1}/{len(questions)}: {question}")
            start = time.time()
            response = await self.ask_question_stream(question)
            elapsed = time.time() - start
            logger.info(f"Got streaming response in {elapsed:.2f} seconds")
            responses.append(response)

        total_time = time.time() - total_time_start
        logger.info(
            f"Asked {len(questions)} streaming questions in {total_time:.2f} seconds (avg: {total_time / len(questions):.2f}s)")
        return responses

    async def ask_questions_direct_stream_sequence(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Ask a sequence of questions with DIRECT AGENT streaming (NEW!).
        
        Args:
            questions: List of natural language questions
            
        Returns:
            List of response data dictionaries
        """
        responses = []
        total_time_start = time.time()

        for i, question in enumerate(questions):
            logger.info(f"Asking DIRECT STREAM question {i + 1}/{len(questions)}: {question}")
            start = time.time()
            response = await self.test_direct_agent_stream(question)
            elapsed = time.time() - start
            logger.info(f"Got direct stream response in {elapsed:.2f} seconds")
            responses.append(response)

        total_time = time.time() - total_time_start
        logger.info(
            f"Asked {len(questions)} direct stream questions in {total_time:.2f} seconds (avg: {total_time / len(questions):.2f}s)")
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
    parser = argparse.ArgumentParser(description="Test backend performance with STREAMING support")
    parser.add_argument("--url", default=DEFAULT_BACKEND_URL, help="Backend URL")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="User ID for testing")
    parser.add_argument("--study-group", default="interactive", choices=["interactive", "chat", "static"],
                        help="Study group")
    parser.add_argument("--ml-knowledge", default="low", choices=["low", "medium", "high"], help="ML knowledge level")
    parser.add_argument("--prediction", default=">50K", help="User prediction to set")
    parser.add_argument("--questions-file", help="JSON file with questions to ask")
    parser.add_argument("--use-stream", action="store_true", help="Use streaming instead of regular requests")
    parser.add_argument("--use-direct-stream", action="store_true", help="Use DIRECT AGENT streaming (bypasses Flask)")
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend initialization (for direct streaming only)")
    args = parser.parse_args()

    # Initialize backend mocker
    mocker = BackendMocker(args.url, args.user_id)

    try:
        # Load questions from file or use default questions
        if args.questions_file:
            with open(args.questions_file, 'r') as f:
                questions = json.load(f)
        else:
            questions = [
                "Why did the model predict this way?",
                "What are the most important features?",
                "How confident is the model?",
            ]

        # Handle direct streaming mode (bypasses Flask entirely)
        if args.use_direct_stream:
            logger.info("🚀 Using DIRECT AGENT STREAMING mode (bypassing Flask backend)")
            logger.info("This tests the new streaming functionality in mape_k_mixins.py directly!")
            
            responses = await mocker.ask_questions_direct_stream_sequence(questions)
            
            # Print summary of responses
            logger.info(f"Asked {len(questions)} direct stream questions, got {len(responses)} responses")
            for i, (question, response) in enumerate(zip(questions, responses)):
                if "error" in response:
                    logger.error(f"Q{i + 1}: '{question}' -> ERROR: {response['error']}")
                else:
                    text = response.get('text', '')[:100] + '...' if len(response.get('text', '')) > 100 else response.get('text', '')
                    elapsed = response.get('elapsed_time', 0)
                    chunks = response.get('chunk_count', 0)
                    first_chunk = response.get('first_chunk_time', 0)
                    timeline = response.get('timing_summary', 'N/A')
                    
                    logger.info(f"Q{i + 1}: '{question}'")
                    logger.info(f"     ⚡ First Chunk: {first_chunk:.2f}s | Total: {elapsed:.2f}s | Chunks: {chunks}")
                    logger.info(f"     📊 Timeline: {timeline}")
                    logger.info(f"     💬 Response: '{text}'")
                    logger.info("")  # Empty line for readability
            
            return  # Skip backend initialization and cleanup

        # Standard backend testing
        if not args.skip_backend:
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

        # Ask questions - choose streaming or regular based on argument
        if args.use_stream:
            logger.info("Using STREAMING mode (via Flask backend)")
            responses = await mocker.ask_questions_stream_sequence(questions)
        else:
            logger.info("Using REGULAR mode")
            responses = await mocker.ask_questions_in_sequence(questions)

        # Print summary of responses
        logger.info(f"Asked {len(questions)} questions, got {len(responses)} responses")
        for i, (question, response) in enumerate(zip(questions, responses)):
            text = response.get('text', '')[:100] + '...' if len(response.get('text', '')) > 100 else response.get(
                'text', '')
            logger.info(f"Q{i + 1}: '{question}' -> '{text}'")

        # Finish experiment
        if not args.skip_backend:
            mocker.finish()

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Try to clean up
        try:
            if not args.skip_backend:
                mocker.finish()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())
