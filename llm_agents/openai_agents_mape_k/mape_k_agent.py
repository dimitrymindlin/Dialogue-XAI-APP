import os
import json
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv
from openai import OpenAI

from agents import Agent, Runner, function_tool
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from create_experiment_data.instance_datapoint import InstanceDatapoint
from llm_agents.merged_prompts import get_merged_prompts

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@function_tool
def get_mape_k_response(user_message: str, 
                        domain_description: str, 
                        feature_names: str,
                        instance_features: Dict[str, Any],
                        predicted_class_name: str,
                        user_model: Dict[str, Any],
                        explanation_collection: Dict[str, Any],
                        chat_history: str,
                        last_shown_explanations: List[Dict[str, Any]] = None,
                        explanation_plan: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process the user message using the MAPE-K workflow and generate a response.
    
    Args:
        user_message: The user's message
        domain_description: Description of the domain
        feature_names: Names of features in the model
        instance_features: Current instance features as a dictionary
        predicted_class_name: The predicted class name
        user_model: The current user model state
        explanation_collection: Collection of available explanations
        chat_history: History of the conversation
        last_shown_explanations: List of explanations that were last shown to the user
        explanation_plan: Current explanation plan
        
    Returns:
        Dictionary containing the MAPE-K workflow results
    """
    # This function is a schema definition for the agent
    # The actual processing will be done in the agent implementation
    return {
        "Monitor": {
            "understanding_displays": ["asking_questions"],
            "cognitive_state": "active"
        },
        "Analyze": {
            "updated_explanation_states": {}
        },
        "Plan": {
            "next_explanations": [
                {
                    "name": "ExplanationName",
                    "description": "Reason for choosing this explanation",
                    "dependencies": [],
                    "is_optional": False
                }
            ],
            "reasoning": "Reasoning behind the chosen explanations"
        },
        "Execute": {
            "html_response": "<p>HTML formatted response</p>"
        }
    }

class MapeKAgent(XAIBaseAgent):
    """
    MAPE-K agent implementation using OpenAI Agents SDK with the unified prompt.
    """
    def __init__(
            self,
            feature_names="",
            domain_description="",
            user_ml_knowledge="",
            experiment_id="",
            verbose=False,
            **kwargs
    ):
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.user_ml_knowledge = user_ml_knowledge
        self.experiment_id = experiment_id
        self.verbose = verbose
        
        # Initialize state
        self.instance = None
        self.predicted_class_name = None
        self.opposite_class_name = None
        self.visual_explanations_dict = {}
        self.datapoint_count = 0
        self.chat_history = "No history available, beginning of the chat."
        self.user_model = {
            "User Info": {
                "ML Knowledge": f"{user_ml_knowledge.capitalize()}, which means that the user may have a basic understanding of AI from casual exposure or news reports. Explanations should focus on simple ideas and relatable examples, avoiding technical jargon or complex interpretability methods.",
                "Cognitive State": "",
                "Explicit Understanding Signals": []
            },
            "NOT_YET_EXPLAINED": {
                "Counterfactuals": [("Concept", []), ("ImpactMultipleFeatures", []), ("ImpactSingleFeature", [])],
                "FeatureImportances": [("Concept", []), ("FeatureInfluencesPlot", []), ("FeaturesInFavourOfOver50k", []), ("FeaturesInFavourOfUnder50k", []), ("WhyThisFeatureImportant", [])],
                "AnchorExplanation": [("Concept", []), ("Anchor", [])],
                "FeatureStatistics": [("Concept", []), ("Feature Statistics", [])],
                "TextualPartialDependence": [("Concept", []), ("PDPDescription", [])],
                "PossibleClarifications": [("Concept", []), ("ClarificationsList", [])],
                "ModelPredictionConfidence": [("Concept", []), ("Confidence", [])],
                "CeterisParibus": [("Concept", []), ("PossibleClassFlips", []), ("ImpossibleClassFlips", [])]
            }
        }
        self.explanation_collection = {}
        self.last_shown_explanations = []
        self.explanation_plan = []
        
        # Create OpenAI agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create an OpenAI Agent with appropriate instructions and tools."""
        # Get the unified prompt
        unified_prompt = get_merged_prompts()
        
        # Create agent
        mape_k_agent = Agent(
            name="MAPE-K XAI Assistant",
            instructions=unified_prompt,
            tools=[get_mape_k_response]
        )
        
        return mape_k_agent
    
    def append_to_history(self, role, msg):
        """Append a message to the chat history."""
        if role == "user":
            msg = "User: " + msg + "\n"
        elif role == "agent":
            msg = "Agent: " + msg + "\n"

        if self.chat_history == "No history available, beginning of the chat.":
            self.chat_history = msg
        else:
            self.chat_history += msg
    
    def initialize_new_datapoint(self,
                                 instance: InstanceDatapoint,
                                 xai_explanations,
                                 xai_visual_explanations,
                                 predicted_class_name,
                                 opposite_class_name,
                                 datapoint_count):
        """Initialize a new datapoint for analysis."""
        self.instance = instance.displayable_features
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.datapoint_count = datapoint_count + 1
        self.visual_explanations_dict = xai_visual_explanations
        self.explanation_collection = xai_explanations
        
        # Reset chat history for new datapoint
        self.chat_history = "No history available, beginning of the chat."
        
        # Reset explanation plan and last shown explanations
        self.explanation_plan = []
        self.last_shown_explanations = []
        
        if self.verbose:
            print(f"MapeKAgent initialized with datapoint {self.datapoint_count}")
    
    async def answer_user_question(self, user_question):
        """Answer a user question using the MAPE-K workflow."""
        try:
            # Run the agent with the merged prompt
            result = await Runner.run(
                self.agent, 
                input={
                    "user_message": user_question,
                    "domain_description": self.domain_description,
                    "feature_names": self.feature_names,
                    "instance_features": self.instance,
                    "predicted_class_name": self.predicted_class_name,
                    "user_model": self.user_model,
                    "explanation_collection": self.explanation_collection,
                    "chat_history": self.chat_history,
                    "last_shown_explanations": self.last_shown_explanations,
                    "explanation_plan": self.explanation_plan
                }
            )
            
            # Extract the MAPE-K result
            mape_k_result = result.outputs.get("get_mape_k_response", {})
            
            if not mape_k_result:
                # Fallback to direct API call if function calling fails
                return await self._direct_mape_k_call(user_question)
            
            # Extract components
            monitor_result = mape_k_result.get("Monitor", {})
            analyze_result = mape_k_result.get("Analyze", {})
            plan_result = mape_k_result.get("Plan", {})
            execute_result = mape_k_result.get("Execute", {})
            
            # Update user model
            understanding_displays = monitor_result.get("understanding_displays", [])
            cognitive_state = monitor_result.get("cognitive_state", "")
            
            if cognitive_state:
                self.user_model["User Info"]["Cognitive State"] = cognitive_state
            
            if understanding_displays:
                self.user_model["User Info"]["Explicit Understanding Signals"] = understanding_displays
            
            # Update explanation states
            updated_states = analyze_result.get("updated_explanation_states", {})
            for exp_name, new_state in updated_states.items():
                if exp_name in self.user_model.get("NOT_YET_EXPLAINED", {}):
                    # Move the explanation to the appropriate state category
                    explanation_info = self.user_model["NOT_YET_EXPLAINED"].pop(exp_name, None)
                    if explanation_info:
                        if new_state not in self.user_model:
                            self.user_model[new_state] = {}
                        self.user_model[new_state][exp_name] = explanation_info
            
            # Update explanation plan
            self.explanation_plan = plan_result.get("next_explanations", [])
            
            # Get the HTML response
            response = execute_result.get("html_response", "")
            
            # Replace placeholders with visual content if available
            for exp_name, exp_data in self.visual_explanations_dict.items():
                placeholder = f"##{exp_name}##"
                if placeholder in response and isinstance(exp_data, str):
                    response = response.replace(placeholder, exp_data)
            
            # Update chat history
            self.append_to_history("user", user_question)
            self.append_to_history("agent", response)
            
            # Generate reasoning
            reasoning = f"Monitor: {json.dumps(monitor_result)}\nAnalyze: {json.dumps(analyze_result)}\nPlan: {plan_result.get('reasoning', '')}"
            
            # Update last shown explanations based on plan
            for explanation in self.explanation_plan:
                name = explanation.get("name")
                if name:
                    self.last_shown_explanations.append({
                        "explanation_name": name,
                        "step_name": explanation.get("dependencies", [""])[0] if explanation.get("dependencies") else ""
                    })
            
            return reasoning, response
            
        except Exception as e:
            print(f"Error in MAPE-K agent: {e}")
            return "Error processing request", "<p>I encountered an error processing your request. Please try again.</p>"
    
    async def _direct_mape_k_call(self, user_question):
        """Direct API call as a fallback if function calling fails."""
        try:
            # Format the prompt with necessary context
            formatted_prompt = get_merged_prompts().format(
                domain_description=self.domain_description,
                feature_names=self.feature_names,
                instance=json.dumps(self.instance) if self.instance else "{}",
                predicted_class_name=self.predicted_class_name if self.predicted_class_name else "Not available",
                explanation_collection=json.dumps(self.explanation_collection) if self.explanation_collection else "{}",
                chat_history=self.chat_history,
                user_message=user_question,
                user_model=json.dumps(self.user_model) if self.user_model else "{}",
                last_shown_explanations=json.dumps(self.last_shown_explanations) if self.last_shown_explanations else "[]",
                explanation_plan=json.dumps(self.explanation_plan) if self.explanation_plan else "[]",
                understanding_displays="[\"asking_questions\", \"seeking_clarification\", \"confirming_understanding\", \"expressing_confusion\", \"expressing_disagreement\", \"proposing_alternative\", \"making_suggestion\", \"comparing_to_prior_knowledge\", \"expressing_surprise\", \"requesting_specific_information\"]",
                modes_of_engagement="\"active\" (straightforward participation like repetition or acknowledgment without elaboration), \"constructive\" (contributes new ideas, generates alternatives), \"interactive\" (builds on conversation, adapts to interactions with system)"
            )
            
            # Call OpenAI API directly
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an XAI assistant that follows the MAPE-K workflow."},
                    {"role": "user", "content": formatted_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            result_text = response.choices[0].message.content
            mape_k_result = json.loads(result_text)
            
            # Extract the HTML response
            html_response = mape_k_result.get("Execute", {}).get("html_response", "")
            if not html_response:
                html_response = "<p>I'm having trouble generating a response. Could you please rephrase your question?</p>"
            
            # Generate reasoning
            reasoning = f"Direct API call used as fallback. Plan reasoning: {mape_k_result.get('Plan', {}).get('reasoning', 'No reasoning available')}"
            
            # Update chat history
            self.append_to_history("user", user_question)
            self.append_to_history("agent", html_response)
            
            return reasoning, html_response
            
        except Exception as e:
            print(f"Error in direct MAPE-K call: {e}")
            return "Error in direct API call", "<p>I encountered a technical issue. Please try again later.</p>"

# Test function
async def test_mape_k_agent():
    """Test the MAPE-K agent with the unified prompt."""
    # Create a mock instance
    class MockInstance:
        def __init__(self):
            self.displayable_features = {
                "age": 45,
                "education": "Bachelors",
                "occupation": "Executive-Managerial",
                "hours_per_week": 60
            }
            self.instance_id = "test_instance"
    
    # Initialize agent
    agent = MapeKAgent(
        feature_names="age, education, occupation, hours_per_week",
        domain_description="An income prediction model that determines if a person earns above or below $50K per year",
        user_ml_knowledge="beginner",
        experiment_id="test",
        verbose=True
    )
    
    # Initialize a new datapoint
    agent.initialize_new_datapoint(
        instance=MockInstance(),
        xai_explanations={"SHAP": {"age": 0.3, "education": 0.5}},
        xai_visual_explanations={"FeatureInfluencesPlot": "<img src='plot_data' />"},
        predicted_class_name=">50K",
        opposite_class_name="<=50K",
        datapoint_count=0
    )
    
    # Test questions
    questions = [
        "How does this model work?",
        "Why did the model predict >50K for this person?",
        "Which features are most important?",
        "How does education affect the prediction?"
    ]
    
    for question in questions:
        print(f"\n\nQuestion: {question}")
        reasoning, response = await agent.answer_user_question(question)
        print(f"Response: {response}")
        print(f"Reasoning: {reasoning}")

if __name__ == "__main__":
    asyncio.run(test_mape_k_agent()) 