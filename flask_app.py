"""The app main."""
import json
import logging
import os
import traceback
import time
from datetime import datetime
from functools import wraps
import base64

from flask import Flask
from flask import request, Blueprint
from flask import jsonify, Response
from flask_cors import CORS
import gin
from dotenv import load_dotenv
import openai
import matplotlib

from explain.logic import ExplainBot


# gunicorn doesn't have command line flags, using a gin file to pass command line args
@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl, use_llm_agent="unified_mape_k"):
        self.config = config
        self.baseurl = baseurl
        self.use_llm_agent = use_llm_agent


# Parse gin global config
gin.parse_config_file("global_config.gin")

# Get args
args = GlobalArgs()

# Komut satırından gelen parametreyi gin_config'e aktarma
if args.use_llm_agent != "unified_mape_k":  # varsayılan değer değişmişse
    gin.bind_parameter("ExplainBot.use_llm_agent", args.use_llm_agent)

bp = Blueprint('host', __name__, template_folder='templates')

CORS(bp)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse application level configs
gin.parse_config_file(args.config)

# Setup the explainbot dict to run multiple bots
bot_dict = {}

"""@bp.route('/')
def home():
    # Load the explanation interface.
    user_id = request.args.get("user_id")
    study_group = request.args.get("study_group")
    if user_id is None:
        user_id = "TEST"
    BOT = ExplainBot(study_group)
    bot_dict[user_id] = BOT
    app.logger.info("Loaded Login and created bot")
    objective = bot_dict[user_id].conversation.describe.get_dataset_objective()
    return render_template("index.html", currentUserId=user_id, datasetObjective=objective)"""


@bp.route('/init', methods=['GET'])
def init():
    """Initialize the bot."""
    user_id = request.args.get('user_id')
    study_group = request.args.get('study_group', "compare")
    ml_knowledge = request.args.get('ml_knowledge', "")
    log_message = f"Creating bot with {study_group}, {ml_knowledge}"
    print(log_message)
    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400

    # API isteği ile gelen agent_type değeri varsa, gin.bind_parameter ile geçici olarak değiştir
    agent_type = request.args.get('use_llm_agent', None)
    if agent_type and agent_type != 'false':
        gin.bind_parameter("ExplainBot.use_llm_agent", agent_type)
    elif agent_type == 'false':
        gin.bind_parameter("ExplainBot.use_llm_agent", False)
        
    # Create the bot - Parametreler Gin config'ten otomatik olarak enjekte edilecek
    bot = ExplainBot(study_group=study_group,
                    ml_knowledge=ml_knowledge,
                    user_id=user_id)

    # Store the bot 
    bot_dict[user_id] = bot
    # Return success
    return jsonify({"status": "OK", "message": "Bot initialized"}), 200


@bp.route('/finish', methods=['DELETE'])
def finish():
    """
    Finish the experiment.
    """
    user_id = request.args.get("user_id")
    if user_id is None:
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
    # Önce user_id kontrolü
    if not user_id:
        raise ValueError("No user_id provided")
    
    if user_id not in bot_dict:
        raise ValueError(f"User ID '{user_id}' not initialized")
    
    # convert to 0-indexed count
    try:
        datapoint_count = int(datapoint_count) - 1
    except (TypeError, ValueError):
        raise ValueError(f"Invalid datapoint_count: {datapoint_count}")

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
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
    
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
    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
        
    datapoint_count = request.args.get("datapoint_count")
    return get_datapoint(user_id, "test", datapoint_count)


@bp.route('/get_final_test_datapoint', methods=['GET'])
def get_final_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
        
    datapoint_count = request.args.get("datapoint_count")
    return get_datapoint(user_id, "final-test", datapoint_count)


@bp.route('/get_intro_test_datapoint', methods=['GET'])
def get_intro_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
        
    datapoint_count = request.args.get("datapoint_count")
    return get_datapoint(user_id, "intro-test", datapoint_count)


@bp.route("/set_user_prediction", methods=['POST'])
def set_user_prediction():
    """Set the user prediction and get the initial message if in teaching phase."""
    data = request.get_json()  # Get JSON data from request body
    user_id = data.get("user_id")
    
    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404

    experiment_phase = data.get("experiment_phase")
    datapoint_count = int(data.get("datapoint_count")) - 1  # 0 indexed for backend
    user_prediction = data.get("user_prediction")
    
    bot = bot_dict[user_id]
    if experiment_phase == "teaching":  # Called differently in the frontend
        experiment_phase = "train"

    user_correct, correct_prediction_string = bot.set_user_prediction(experiment_phase,
                                                                      datapoint_count,
                                                                      user_prediction)

    # If not in teaching phase, return 200 OK
    if experiment_phase != "train":
        return jsonify({"message": "OK"}), 200
    else:
        # Create initial message depending on the user study group and whether the user was correct
        user_study_group = bot.get_study_group()
        if user_study_group == "interactive":
            if user_correct:
                prompt = f"""
                    <b>Correct!</b> The model predicted <b>{correct_prediction_string}</b> for the current {bot.instance_type_naming}. <br>
                    The model <b>starts with a 75% chance that the person earns below $50K</b>, based on general trends and then considers
                    the individual's attributes to make a prediction. <br>
                    If you want to <b>verify if your reasoning</b> aligns with the model, <b>select questions</b> from the right.
                    """
            else:
                prompt = f"""
                    Not quite right according to the model… It predicted <b>{correct_prediction_string}</b> for this {bot.instance_type_naming}.
                    The model <b>starts with a 75% chance that the person earns below $50K</b>, based on general trends and then considers
                    the individual's attributes to make a prediction. <br>
                    To <b>understand the model's reasoning</b> and improve your future predictions, <b>select questions</b> from the right.
                    """
        else:  # chat
            if user_correct:
                prompt = f"""
                    <b>Correct!</b> The model predicted <b>{correct_prediction_string}</b>. <br>
                    If you want to <b>verify if your reasoning</b> aligns with the model, <b>type your questions</b> about the model prediction in the chat."""
            else:
                prompt = f"""
                Not quite right according to the model… It predicted <b>{correct_prediction_string}</b> for this {bot.instance_type_naming}. <br>
                To understand its reasoning and improve your predictions, <b>type your questions</b> in the chat, and I will answer them."""
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
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
        
    bot = bot_dict[user_id]
    correctness_string = bot.get_user_correctness()
    response = {"correctness_string": correctness_string}
    return response


@bp.route("/get_proceeding_okay", methods=['GET'])
def get_proceeding_okay():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
        
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
    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
        
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
        return jsonify({"error": "No user_id provided"}), 400
    
    if user_id not in bot_dict:
        return jsonify({"error": f"User ID '{user_id}' not initialized. Call /init first."}), 404
        
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


app = Flask(__name__)
app.register_blueprint(bp, url_prefix=args.baseurl)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# CORS(app, resources={r"/*": {"origins": "http://dialogue-xai-frontend:3000"}})

# Create cache folder in root if it doesn't exist
if not os.path.exists("cache"):
    os.makedirs("cache")

if __name__ != '__main__':
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    app.logger.addHandler(stream_handler)
    app.logger.setLevel(logging.INFO)
    matplotlib.use('Agg')  

if __name__ == "__main__":
    # clean up storage file on restart
    app.logger.info(f"Launching app from config: {args.config}")
    matplotlib.use('Agg') 
    
    app.run(debug=True, port=4555, host='0.0.0.0', use_reloader=False)
