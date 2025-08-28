"""The app main."""
import json
import logging
import os
import traceback
import base64

from flask import Flask
from flask import request, Blueprint
from flask import jsonify, Response
from flask_cors import CORS
import gin
import openai
import matplotlib

from explain.logic import ExplainBot

import mlflow
import mlflow.llama_index

from dotenv import load_dotenv

# Define API blueprint at module level
bp = Blueprint('host', __name__, template_folder='templates')
# Allow CORS for our React frontend
CORS(bp)


@gin.configurable
class GlobalArgs:
    """Global configuration arguments for the application.
    
    This class is configured via gin files and provides command line
    argument functionality for gunicorn deployments.
    """
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl


# Setup the explainbot dict to run multiple bots
bot_dict = {}


def _load_environment():
    """Load environment variables from .env files."""
    load_dotenv()
    
    # Load local environment file if it exists (for development)
    if os.path.exists('.env.local'):
        load_dotenv('.env.local', override=True)


def _configure_gin():
    """Parse and configure gin configuration files."""
    gin.parse_config_file("global_config.gin")
    args = GlobalArgs()
    gin.parse_config_file(args.config)
    return args


def _setup_directories():
    """Create necessary directories for the application."""
    directories = ["cache", "cache/mlruns"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def _get_mlflow_uri():
    """Get and normalize MLflow tracking URI for local and Docker environments."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file://./cache/mlruns")
    
    if mlflow_uri.startswith("file://"):
        # Extract path from file:// URI
        mlflow_path = mlflow_uri.replace("file://", "")
        
        # Convert Docker path to local path when running locally
        if mlflow_path.startswith("/usr/src/app/"):
            mlflow_path = mlflow_path.replace("/usr/src/app/", "./")
        
        # Ensure directory exists
        if not os.path.exists(mlflow_path):
            os.makedirs(mlflow_path, exist_ok=True)
        
        # Update URI to use correct local path if not in Docker
        if not mlflow_uri.startswith("file:///usr/src/app/") or not os.path.exists("/usr/src/app"):
            mlflow_uri = f"file://{os.path.abspath(mlflow_path)}"
    
    return mlflow_uri


def _initialize_mlflow(app):
    """Initialize MLflow tracking."""
    try:
        mlflow_uri = _get_mlflow_uri()
        mlflow.set_tracking_uri(mlflow_uri)
        app.logger.info(f"MLflow tracking URI initialized: {mlflow_uri}")
    except Exception as e:
        app.logger.warning(f"MLflow startup init failed: {e}")


def _configure_logging(app):
    """Configure application logging."""
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    app.logger.addHandler(stream_handler)
    app.logger.setLevel(logging.INFO)


def _set_environment_variables():
    """Set required environment variables."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_app():
    """Create and configure the Flask application.
    
    Returns:
        Flask: Configured Flask application instance.
    """
    # Load environment configuration
    _load_environment()
    
    # Configure gin settings
    args = _configure_gin()
    
    # Create necessary directories
    _setup_directories()
    
    # Initialize Flask app
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
    
    # Register blueprints
    app.register_blueprint(bp, url_prefix=args.baseurl)
    
    # Initialize external services
    _initialize_mlflow(app)
    
    # Configure logging
    _configure_logging(app)
    
    # Configure matplotlib for headless operation
    matplotlib.use('Agg')
    
    # Set environment variables
    _set_environment_variables()
    
    # Make app available to route functions
    globals()["app"] = app
    
    return app


@bp.route('/init', methods=['GET'])
def init():
    """Load the explanation interface."""
    user_id = request.args.get("user_id")
    study_group = request.args.get("study_group")
    ml_knowledge = request.args.get("ml_knowledge")
    if not user_id:
        user_id = "TEST"
    if not study_group:
        study_group = "interactive"
    if not ml_knowledge:
        ml_knowledge = "low"
    BOT = ExplainBot(study_group, ml_knowledge, user_id)
    bot_dict[user_id] = BOT
    app.logger.info("Loaded Login and created bot")

    # Initialize MLflow experiment for this user
    initialize_mlflow_experiment(user_id)
    app.logger.info("MLflow experiment initialized for user: %s", user_id)

    # Feature tooltip and units
    feature_tooltip = bot_dict[user_id].get_feature_tooltips()
    feature_units = bot_dict[user_id].get_feature_units()
    questions = bot_dict[user_id].get_questions_attributes_featureNames()
    ordered_feature_names = bot_dict[user_id].get_feature_names()
    user_experiment_prediction_choices = bot_dict[user_id].conversation.class_names
    user_study_task_description = bot_dict[user_id].conversation.describe.get_user_study_objective()
    result = {
        "feature_tooltips": feature_tooltip,
        "feature_units": feature_units,
        'questions': questions,
        'feature_names': ordered_feature_names,
        'prediction_choices': user_experiment_prediction_choices,
        'user_study_task_description': user_study_task_description
    }
    return result


@bp.route('/finish', methods=['DELETE'])
def finish():
    """
    Finish the experiment.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = "TEST"
    # Remove the bot from the dict
    try:
        bot_dict.pop(user_id)
    except KeyError:
        print(f"User {user_id} sent finish again, but the Bot was not in the dict.")
        return "200 OK"
    print(f"User {user_id} finished the experiment. And the Bot was removed from the dict.")
    return "200 OK"


def get_datapoint(user_id, datapoint_type, datapoint_count, return_probability=False):
    """
    Get a datapoint from the dataset based on the datapoint type.
    """
    # convert to 0-indexed count
    datapoint_count = int(datapoint_count) - 1

    if not user_id:
        user_id = "TEST"
    instance = bot_dict[user_id].get_next_instance(datapoint_type,
                                                   datapoint_count,
                                                   return_probability=return_probability)
    instance_dict = instance.get_datapoint_as_dict_for_frontend()
    return instance_dict


@bp.route('/get_train_datapoint', methods=['GET'])
def get_train_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = "TEST"
    datapoint_count = request.args.get("datapoint_count")
    user_study_group = bot_dict[user_id].get_study_group()
    result_dict = get_datapoint(user_id, "train", datapoint_count)
    if bot_dict[user_id].use_active_dialogue_manager:
        bot_dict[user_id].reset_dialogue_manager()

    if user_study_group == "static":
        # Get the explanation report
        static_report = bot_dict[user_id].get_explanation_report()
        static_report["instance_type"] = bot_dict[user_id].instance_type_naming
        result_dict["static_report"] = static_report
    return result_dict


@bp.route('/get_test_datapoint', methods=['GET'])
def get_test_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    datapoint_count = request.args.get("datapoint_count")
    return get_datapoint(user_id, "test", datapoint_count)


@bp.route('/get_final_test_datapoint', methods=['GET'])
def get_final_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    datapoint_count = request.args.get("datapoint_count")
    return get_datapoint(user_id, "final-test", datapoint_count)


@bp.route('/get_intro_test_datapoint', methods=['GET'])
def get_intro_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    datapoint_count = request.args.get("datapoint_count")
    return get_datapoint(user_id, "intro-test", datapoint_count)


@bp.route("/set_user_prediction", methods=['POST'])
def set_user_prediction():
    """Set the user prediction and get the initial message if in teaching phase."""
    data = request.get_json()  # Get JSON data from request body
    user_id = data.get("user_id")
    experiment_phase = data.get("experiment_phase")
    datapoint_count = int(data.get("datapoint_count")) - 1  # 0 indexed for backend
    user_prediction = data.get("user_prediction")
    if not user_id:
        user_id = "TEST"  # Default user_id for testing
    bot = bot_dict[user_id]
    if experiment_phase == "teaching":  # Called differently in the frontend
        experiment_phase = "train"

    try:
        user_correct, correct_prediction_string = bot.set_user_prediction(experiment_phase,
                                                                          datapoint_count,
                                                                          user_prediction)
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'suggestion': 'Please request the datapoint first by calling the appropriate get_*_datapoint endpoint'
        }), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

    # If not in teaching phase, return 200 OK
    if experiment_phase != "train":
        return jsonify({"message": "OK"}), 200
    else:
        # Create initial message depending on the user study group and whether the user was correct
        user_study_group = bot.get_study_group()
        if user_study_group == "interactive":
            if user_correct:
                prompt = f"""<b>Correct!</b> The model predicted <b>{correct_prediction_string}</b> for the current {bot.instance_type_naming}. <br>The model <b>starts with a 75% chance that the person earns below $50K</b>, based on general trends and then considers the individual's attributes to make a prediction. <br>If you want to <b>verify if your reasoning</b> aligns with the model, <b>select questions</b> from the right."""
            else:
                prompt = f"""Not quite right according to the model… It predicted <b>{correct_prediction_string}</b> for this {bot.instance_type_naming}. The model <b>starts with a 75% chance that the person earns below $50K</b>, based on general trends and then considers the individual's attributes to make a prediction. <br>To <b>understand the model's reasoning</b> and improve your future predictions, <b>select questions</b> from the right."""
        else:  # chat
            if user_correct:
                prompt = f"""<b>Correct!</b> The model predicted <b>{correct_prediction_string}</b>. <br>If you want to <b>verify if your reasoning</b> aligns with the model, <b>type your questions</b> about the model prediction in the chat."""
            else:
                prompt = f"""Not quite right according to the model… It predicted <b>{correct_prediction_string}</b> for this {bot.instance_type_naming}. <br>To understand its reasoning and improve your predictions, <b>type your questions</b> in the chat, and I will answer them."""
            if bot_dict[user_id].use_llm_agent:
                bot_dict[user_id].agent.append_to_history("agent", prompt)

        message = {
            "isUser": False,
            "feedback": False,
            "text": prompt,
            "question_id": "init",
            "feature_id": 0,
            "followup": [],
            "reasoning": ""
        }
        return jsonify({"initial_message": message}), 200


@bp.route("/get_user_correctness", methods=['GET'])
def get_user_correctness():
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = "TEST"
    bot = bot_dict[user_id]
    correctness_string = bot.get_user_correctness()
    response = {"correctness_string": correctness_string}
    return response


@bp.route("/get_proceeding_okay", methods=['GET'])
def get_proceeding_okay():
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = "TEST"
    bot = bot_dict[user_id]
    proceeding_okay, follow_up_questions, response_text = bot.get_proceeding_okay()
    # Make it a message dict
    message = {
        "isUser": False,
        "feedback": True,
        "text": response_text,
        "id": 1000,
        "followup": follow_up_questions,
        "reasoning": "",
    }
    return {"proceeding_okay": proceeding_okay, "message": message}


def generate_audio_from_text(text, voice="alloy"):
    """
    Generate audio from text using OpenAI's TTS API.
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: "alloy")
        
    Returns:
        dict: A dictionary containing the audio data in base64 format or an error message
    """
    try:
        if not openai.api_key:
            app.logger.warning("OpenAI API key is not configured for text-to-speech!")
            return {"error": "OpenAI API key is not configured"}

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        audio_response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Convert audio to base64 for sending in JSON response
        audio_data = audio_response.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        return {
            "data": audio_base64,
            "format": "mp3"
        }
    except Exception as e:
        app.logger.error(f"Error generating speech: {str(e)}")
        app.logger.error(traceback.format_exc())
        return {"error": f"Error generating speech: {str(e)}"}


@bp.route("/get_response_clicked", methods=['POST'])
def get_bot_response():
    """Load the box response."""
    user_id = request.args.get("user_id")
    # bot_dict[user_id].save_all_questions_and_answers_to_csv()
    if not user_id:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the bot response")
        try:
            data = json.loads(request.data)
            question_id = data["question"]
            feature_id = data["feature"]
            response = bot_dict[user_id].update_state_new(question_id=question_id, feature_id=feature_id)
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
            feature_id = None
            question_id = None

        if bot_dict[user_id].use_active_dialogue_manager:
            followup = bot_dict[user_id].get_suggested_method()
        else:
            followup = []
        message_dict = {
            "isUser": False,
            "feedback": True,
            "text": response[0],
            "question_id": question_id,
            "feature_id": feature_id,
            "followup": followup,
            "reasoning": response[3]
        }

        # Check if soundwave parameter is provided in the request
        soundwave = data.get("soundwave", True)
        if soundwave:
            voice = data.get("voice", "alloy")
            audio_result = generate_audio_from_text(response[0], voice)

            if "error" in audio_result:
                message_dict["audio_error"] = audio_result["error"]
            else:
                message_dict["audio"] = audio_result

        return jsonify(message_dict)


@bp.route("/get_response_nl", methods=['POST'])
async def get_bot_response_from_nl():
    """Load the box response."""
    user_id = request.args.get("user_id")
    if not user_id:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the bot response for nl input")
        try:
            data = json.loads(request.data)
            
            # Check if streaming is requested
            enable_streaming = data.get("streaming", False)
            enable_conversational_mode = data.get("show_user_model_panel", False)
            
            if enable_streaming:
                # Redirect to streaming endpoint with same data
                return await get_bot_response_from_nl_stream_internal(user_id, data, enable_conversational_mode)
            
            # Check if bot exists, create if not
            if user_id not in bot_dict:
                app.logger.info(f"Bot not found for user {user_id}, creating new bot")
                from explain.logic import ExplainBot
                BOT = ExplainBot(study_group="chat", ml_knowledge="low", user_id=user_id)
                bot_dict[user_id] = BOT
                app.logger.info(f"Created new bot for user {user_id}")
            
            response, question_id, feature_id, reasoning = await bot_dict[user_id].update_state_from_nl(
                user_input=data["message"])
            if bot_dict[user_id].use_active_dialogue_manager:
                followup = bot_dict[user_id].get_suggested_method()
            else:
                followup = []
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
            question_id = None
            feature_id = None
            followup = []
            reasoning = ""

        assert isinstance(response, str)
        assert isinstance(question_id, int) or question_id is None
        assert isinstance(feature_id, int) or feature_id is None
        assert isinstance(followup, list)
        assert isinstance(reasoning, str)

        message_dict = {
            "isUser": False,
            "feedback": True,
            "text": response,
            "question_id": question_id,
            "feature_id": feature_id,
            "followup": followup,
            "reasoning": reasoning
        }

        # Check if soundwave parameter is provided in the request
        soundwave = data.get("soundwave", True)
        if soundwave:
            voice = data.get("voice", "alloy")
            audio_result = generate_audio_from_text(response, voice)

            if "error" in audio_result:
                message_dict["audio_error"] = audio_result["error"]
            else:
                message_dict["audio"] = audio_result

        return jsonify(message_dict)


async def get_bot_response_from_nl_stream_internal(user_id: str, data: dict, enable_conversational_mode: bool):
    """Internal streaming response handler."""
    user_message = data["message"]
    print("conversational mode", enable_conversational_mode)
    def generate_stream():
        try:
            import asyncio
            import json
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Get the bot
                bot = bot_dict.get(user_id)
                
                # If bot doesn't exist, create it with default settings
                if bot is None:
                    app.logger.info(f"Bot not found for user {user_id}, creating new bot")
                    from explain.logic import ExplainBot
                    BOT = ExplainBot(study_group="chat", ml_knowledge="low", user_id=user_id)
                    bot_dict[user_id] = BOT
                    bot = bot_dict[user_id]
                    app.logger.info(f"Created new bot for user {user_id}")
                
                # Check if the bot has an agent with streaming capability
                if hasattr(bot, 'agent') and hasattr(bot.agent, 'answer_user_question_stream'):
                    app.logger.info("Using agent streaming capability")
                    
                    async def run_streaming():
                        accumulated_response = ""
                        reasoning = ""
                        
                        # Call ExplainBot's streaming method
                        async for chunk in bot.update_state_from_nl_stream(user_message, enable_conversational_mode):
                            
                            if chunk.get("type") == "partial":
                                content = chunk.get("content", "")
                                accumulated_response += content
                                
                                # Send partial chunk to frontend immediately
                                chunk_data = {
                                    "type": "partial",
                                    "content": content,
                                    "is_complete": False
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                            if enable_conversational_mode:
                                if chunk.get("type") == "demographics":
                                    app.logger.info("--- STREAMING DEMOGRAPHICS TO FRONTEND ---")
                                    app.logger.info(json.dumps(chunk, indent=2))
                                    app.logger.info("------------------------------------------")
                                    # Also print user model info derived from get_state_summary -> get_user_info
                                    try:
                                        if hasattr(bot, "agent") and hasattr(bot.agent, "user_model"):
                                            # 1) From get_state_summary(as_dict=True) -> ["User Info"]
                                            state_summary = bot.agent.user_model.get_state_summary(as_dict=True)
                                            user_info = state_summary.get("User Info", {})
                                            app.logger.info("--- USER MODEL INFO (from get_state_summary) ---")
                                            app.logger.info(json.dumps(user_info, ensure_ascii=False, indent=2))
                                            app.logger.info("------------------------------------------------")
                                            # Print to stdout as well
                                            print("--- USER MODEL INFO (from get_state_summary) ---")
                                            print(json.dumps(user_info, ensure_ascii=False, indent=2))
                                            print("------------------------------------------------")

                                            # 2) Directly via get_user_info(as_dict=True)
                                            user_info_direct = bot.agent.user_model.get_user_info(as_dict=True)
                                            app.logger.info("--- USER MODEL INFO (from get_user_info) ---")
                                            app.logger.info(json.dumps(user_info_direct, ensure_ascii=False, indent=2))
                                            app.logger.info("---------------------------------------------")
                                            print("--- USER MODEL INFO (from get_user_info) ---")
                                            print(json.dumps(user_info_direct, ensure_ascii=False, indent=2))
                                            print("---------------------------------------------")

                                            # 3) Extract normalized ML knowledge level and push to frontend in demographics
                                            def _extract_ml_level(val):
                                                allowed = ["very low", "low", "moderate", "high", "very high", "anonymous"]
                                                if val is None:
                                                    return "anonymous"
                                                text = str(val).strip().lower()
                                                # Prefer exact prefix matches for phrases
                                                for phrase in ["very low", "very high"]:
                                                    if text.startswith(phrase):
                                                        return phrase
                                                # Then single-word levels
                                                for lvl in ["very low", "low", "moderate", "high", "very high", "anonymous"]:
                                                    if text.startswith(lvl):
                                                        return lvl
                                                # Fallback: search anywhere in text
                                                for lvl in allowed:
                                                    if lvl in text:
                                                        return lvl
                                                return "anonymous"

                                            ml_value = user_info_direct.get("ML Knowledge") if isinstance(user_info_direct, dict) else None
                                            ml_level = _extract_ml_level(ml_value)
                                            try:
                                                if isinstance(chunk.get("content"), dict):
                                                    chunk["content"]["ml_knowledge"] = ml_level
                                                    app.logger.info(f"Updated demographics ml_knowledge to '{ml_level}' from user model.")
                                            except Exception:
                                                pass
                                        else:
                                            app.logger.info("(User Model) Agent or user_model not available on bot; skipping print.")
                                    except Exception as e:
                                        app.logger.error(f"Error while printing user model info: {str(e)}")
                                    yield f"data: {json.dumps(chunk)}\n\n"
                            
                            elif chunk.get("type") == "final":
                                reasoning = chunk.get("reasoning", "")
                                final_response = chunk.get("content", accumulated_response)
                                question_id = chunk.get("question_id")
                                feature_id = chunk.get("feature_id")
                                
                                # Get followup suggestions if available
                                followup = []
                                if bot.use_active_dialogue_manager:
                                    try:
                                        followup = bot.get_suggested_method()
                                    except:
                                        followup = []
                                
                                # Send final response
                                final_data = {
                                    "type": "final",
                                    "content": final_response,
                                    "reasoning": reasoning,
                                    "followup": followup,
                                    "is_complete": True,
                                    "isUser": False,
                                    "feedback": True,
                                    "question_id": question_id,
                                    "feature_id": feature_id
                                }
                                
                                # Generate audio if requested
                                soundwave = data.get("soundwave", True)
                                if soundwave:
                                    voice = data.get("voice", "alloy")
                                    audio_result = generate_audio_from_text(final_response, voice)
                                    
                                    if "error" in audio_result:
                                        final_data["audio_error"] = audio_result["error"]
                                    else:
                                        final_data["audio"] = audio_result
                                
                                yield f"data: {json.dumps(final_data)}\n\n"
                                
                    
                    # Run the async streaming function
                    async_gen = run_streaming()
                    try:
                        while True:
                            chunk = loop.run_until_complete(async_gen.__anext__())
                            yield chunk
                    except StopAsyncIteration:
                        pass
                
                else:
                    # Fallback to non-streaming response
                    app.logger.info("No streaming capability, using fallback")
                    
                    async def run_fallback():
                        try:
                            response, question_id, feature_id, reasoning = await bot.update_state_from_nl(
                                user_input=user_message)
                            
                            followup = []
                            if bot.use_active_dialogue_manager:
                                try:
                                    followup = bot.get_suggested_method()
                                except:
                                    followup = []
                            
                            final_data = {
                                "type": "final",
                                "content": response,
                                "reasoning": reasoning,
                                "followup": followup,
                                "is_complete": True,
                                "isUser": False,
                                "feedback": True,
                                "question_id": question_id,
                                "feature_id": feature_id
                            }
                            
                            return final_data
                        except Exception as e:
                            app.logger.error(f"Fallback error: {str(e)}")
                            return {
                                "type": "error",
                                "content": "Sorry! I couldn't understand that. Could you please try to rephrase?",
                                "is_complete": True
                            }
                    
                    try:
                        result = loop.run_until_complete(run_fallback())
                        yield f"data: {json.dumps(result)}\n\n"
                    except Exception as e:
                        app.logger.error(f"Fallback execution error: {str(e)}")
                        error_data = {
                            "type": "error",
                            "content": "Sorry! I couldn't understand that. Could you please try to rephrase?",
                            "is_complete": True
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
            
            finally:
                loop.close()
                
        except Exception as e:
            app.logger.error(f"Streaming error: {str(e)}")
            app.logger.error(traceback.format_exc())
            error_data = {
                "type": "error",
                "content": "Sorry! I couldn't understand that. Could you please try to rephrase?",
                "error": str(e),
                "is_complete": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(
        generate_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )


@bp.route("/update_demographics", methods=['POST'])
def update_demographics():
    """Update user demographics based on user input."""
    data = request.get_json()
    user_id = data.get("user_id")
    if user_id is None:
        user_id = "TEST"
    
    demographics_data = data.get("demographics")
    if not demographics_data:
        return jsonify({"error": "No demographics data provided"}), 400
        
    try:
        from llm_agents.models import UserDemographics
        
        # Check if bot exists
        if user_id not in bot_dict:
            return jsonify({"error": f"Bot not found for user {user_id}"}), 404
            
        bot = bot_dict[user_id]
        
        # Validate and update demographics
        user_demographics = UserDemographics(**demographics_data)
        bot.agent.set_user_demographics(user_demographics)
        
        app.logger.info(f"Updated demographics for user {user_id}")
        return jsonify({"message": "Demographics updated successfully"}), 200
        
    except Exception as e:
        app.logger.error(f"Error updating demographics: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Error updating demographics: {str(e)}"}), 500


@bp.route("/get_response_nl_stream", methods=['POST'])
async def get_bot_response_from_nl_stream():
    """Stream the bot response using Server-Sent Events."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    
    data = json.loads(request.data)
    app.logger.info(f"Starting streaming response for user {user_id}")
    
    return await get_bot_response_from_nl_stream_internal(user_id, data)


@bp.route("/update_ml_knowledge", methods=['POST', 'OPTIONS'])
def update_ml_knowledge():
    """Update the ML knowledge level for the current user's user model.

    Body JSON:
        {
          "user_id": "TEST",            # optional if provided as query param, defaults to TEST
          "ml_knowledge": "high"        # one of: very low, low, moderate, high, very high, anonymous
        }

    Returns:
        { "message": str, "ml_knowledge": str }
    """
    try:
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            resp = jsonify({"message": "ok"})
            resp.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            return resp, 200

        data = request.get_json() or {}
        user_id = request.args.get("user_id") or data.get("user_id") or "TEST"
        new_value = data.get("ml_knowledge")

        if not new_value:
            return jsonify({"error": "'ml_knowledge' is required"}), 400

        # Normalize incoming value
        def _normalize_level(val: str) -> str:
            text = str(val).strip().lower().replace("_", " ").replace("-", " ")
            if text.startswith("very low"):  # keep phrase priority
                return "very low"
            if text.startswith("very high"):
                return "very high"
            for lvl in ["very low", "low", "moderate", "high", "very high", "anonymous"]:
                if text == lvl or text.startswith(lvl):
                    return lvl
            # fallback
            return "anonymous"

        normalized = _normalize_level(new_value)

        # Ensure bot exists
        if user_id not in bot_dict:
            return jsonify({"error": f"Bot not found for user {user_id}"}), 404

        bot = bot_dict[user_id]

        # Read previous level (if possible)
        prev_level = None
        try:
            if hasattr(bot, "agent") and hasattr(bot.agent, "user_model"):
                prev_info = bot.agent.user_model.get_user_info(as_dict=True)
                prev_text = (prev_info or {}).get("ML Knowledge")
                # Extract display level from description text
                def _extract_ml_level(val):
                    allowed = ["very low", "low", "moderate", "high", "very high", "anonymous"]
                    if val is None:
                        return "anonymous"
                    t = str(val).strip().lower()
                    for phrase in ["very low", "very high"]:
                        if t.startswith(phrase):
                            return phrase
                    for lvl in allowed:
                        if t.startswith(lvl) or lvl in t:
                            return lvl
                    return "anonymous"
                prev_level = _extract_ml_level(prev_text)
        except Exception:
            prev_level = None

        # Update user model values
        updated_level_desc = None
        if hasattr(bot, "agent") and hasattr(bot.agent, "user_model"):
            try:
                # Update BaseAgent attribute for consistency
                bot.agent.user_ml_knowledge = normalized
                # Update ExplainBot stored level if present
                try:
                    bot.ml_knowledge = normalized
                except Exception:
                    pass
                # Update the actual user model description field using its helper
                updated_level_desc = bot.agent.user_model.set_user_ml_knowledge(normalized)
                bot.agent.user_model.user_ml_knowledge = updated_level_desc
            except Exception as e:
                app.logger.error(f"Failed updating user model ml_knowledge: {e}")
                return jsonify({"error": "Failed to update user model"}), 500
        else:
            # No agent present — best effort update on bot only
            try:
                bot.ml_knowledge = normalized
            except Exception:
                pass

        # Logs
        app.logger.info(
            f"ML knowledge change request for user {user_id}: {prev_level or 'unknown'} -> {normalized}"
        )
        print(f"[update_ml_knowledge] user={user_id} prev={prev_level} new={normalized}")

        resp = jsonify({
            "message": "ML knowledge updated",
            "ml_knowledge": normalized
        })
        resp.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        resp.headers['Vary'] = 'Origin'
        return resp, 200

    except Exception as e:
        app.logger.error(f"Error updating ML knowledge: {str(e)}")
        app.logger.error(traceback.format_exc())
        resp = jsonify({"error": f"Error updating ML knowledge: {str(e)}"})
        resp.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        resp.headers['Vary'] = 'Origin'
        return resp, 500

@bp.route("/speech-to-text", methods=['POST'])
async def transcribe_audio():
    """
    Endpoint to convert speech to text using OpenAI's Whisper API.
    Accepts an audio file and returns the transcribed text.
    """
    try:
        user_id = request.form.get("user_id")
        audio_file = request.files.get("audio_file")

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        # Save the uploaded file temporarily
        temp_file_path = f"temp_{audio_file.filename}"
        audio_file.save(temp_file_path)

        try:
            # Call OpenAI's API to transcribe the audio

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with open(temp_file_path, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )

            # Return the transcribed text
            return jsonify({
                "text": transcript.text,
                "user_id": user_id
            })

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error transcribing audio: {str(e)}"}), 500


@bp.route("/text-to-speech", methods=['POST'])
async def text_to_speech():
    """
    Endpoint to convert text to speech using OpenAI's TTS API.
    Accepts text input and returns audio stream.
    """
    try:
        data = json.loads(request.data)
        text = data.get("text")
        voice = data.get("voice", "alloy")  # Default voice is alloy

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Check if OpenAI API key is configured
        if not openai.api_key:
            print("OpenAI API key is not configured!")
            return jsonify({"error": "API key is not configured"}), 500

        # Call OpenAI's API to generate speech
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Create a streaming response
        def generate():
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

        # Return the audio stream
        return Response(generate(), mimetype="audio/mpeg")

    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error generating speech: {str(e)}"}), 500


@bp.route("/test-tts", methods=['GET'])
def test_tts_page():
    """
    Serve the test TTS HTML page.
    """
    return app.send_static_file('test_tts.html')


def initialize_mlflow_experiment(user_id):
    """Initialize MLflow experiment for a specific user."""
    try:
        # Ensure user_id is valid and not empty
        if not user_id or user_id.strip() == "":
            user_id = "DEFAULT_USER"
        
        experiment_name = f"{user_id}"
        
        # Set the experiment (creates it if it doesn't exist)
        mlflow.set_experiment(experiment_name)
        
        # mlflow.openai.autolog(log_traces=True)
        mlflow.llama_index.autolog(log_traces=True)
        app.logger.info(f"MLflow experiment '{experiment_name}' initialized for user {user_id}.")
        return True
    except Exception as e:
        app.logger.warning(f"MLflow experiment init failed for user {user_id}: {e}")
        app.logger.info("Continuing without MLflow tracking. App will function normally.")
        return False


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=4555, host='0.0.0.0', use_reloader=False)
