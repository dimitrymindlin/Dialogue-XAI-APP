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
    """Always point MLflow to ./cache/mlruns (absolute)."""
    target = os.path.abspath(os.path.join("cache", "mlruns"))
    os.makedirs(target, exist_ok=True)
    return f"file://{target}"


def _initialize_mlflow(app):
    """Initialize MLflow tracking."""
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", _get_mlflow_uri())
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

    # Suppress matplotlib font manager DEBUG messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

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

    # Update MLflow experiment for the current chat round
    if datapoint_count:
        update_mlflow_experiment_for_round(user_id, int(datapoint_count))

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

        # Generate dataset-dependent baseline probability text
        baseline_prob_text = bot.generate_baseline_probability_text()

        if user_study_group == "interactive":
            if user_correct:
                prompt = f"""<b>Correct!</b> The model predicted <b>{correct_prediction_string}</b> for the current {bot.instance_type_naming}. <br>{baseline_prob_text} <br>If you want to <b>verify if your reasoning</b> aligns with the model, <b>select questions</b> from the right."""
            else:
                prompt = f"""Not quite right according to the model… It predicted <b>{correct_prediction_string}</b> for this {bot.instance_type_naming}. {baseline_prob_text} <br>To <b>understand the model's reasoning</b> and improve your future predictions, <b>select questions</b> from the right."""
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


def initialize_mlflow_experiment(user_id, datapoint_count=None):
    """Initialize MLflow experiment for a specific user and chat round."""
    try:
        if user_id == "TEST" and not datapoint_count:  # for testing purposes
            datapoint_count = 0

        # Create hierarchical experiment name including chat round
        if datapoint_count is not None:
            # Format: USERID_N (e.g., user_998852294013090438_0)
            experiment_name = f"{user_id}_{datapoint_count}"
        else:
            # Fallback for initialization without datapoint_count
            experiment_name = f"{user_id}_session"

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


def update_mlflow_experiment_for_round(user_id, datapoint_count):
    """Update MLflow experiment when moving to a new chat round."""
    datapoint_count -= 1  # Convert to 0-indexed count
    return initialize_mlflow_experiment(user_id, datapoint_count)


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
