"""The app main."""
import json
import logging
import os
import traceback

from flask import Flask
from flask import render_template, request, Blueprint
from flask_cors import CORS
import gin

from explain.logic import ExplainBot


# gunicorn doesn't have command line flags, using a gin file to pass command line args
@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl


# Parse gin global config
gin.parse_config_file("global_config.gin")

# Get args
args = GlobalArgs()

bp = Blueprint('host', __name__, template_folder='templates')

CORS(bp)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse application level configs
gin.parse_config_file(args.config)

# Setup the explainbot dict to run multiple bots
bot_dict = {}


# BOT.build_exit_survey_table()


@bp.route('/')
def home():
    """Load the explanation interface."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    BOT = ExplainBot(user_id)
    bot_dict[user_id] = BOT
    app.logger.info("Loaded Login and created bot")
    objective = bot_dict[user_id].conversation.describe.get_dataset_objective()
    return render_template("new.html", currentUserId=user_id, datasetObjective=objective)


@bp.route('/init', methods=['GET'])
def init():
    """Load the explanation interface."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    BOT = ExplainBot(user_id)
    bot_dict[user_id] = BOT
    app.logger.info("Loaded Login and created bot")
    # Questions
    questions = bot_dict[user_id].get_questions_and_attributes()
    # Feature tooltip
    feature_tooltip = bot_dict[user_id].get_feature_tooltips()
    result = {
        "questions": questions,
        "feature_tooltip": feature_tooltip
    }
    return result


@bp.route('/get_train_datapoint', methods=['GET'])
def get_train_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    instance_id, instance_dict, prediction_proba = bot_dict[user_id].get_next_instance()
    instance_dict["id"] = str(instance_id)
    # Make sure all values are strings
    for key, value in instance_dict.items():
        # turn floats to strings if float has zeroes after decimal point
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        instance_dict[key] = str(value)

    # Get initial prompt
    current_prediction = bot_dict[user_id].get_current_prediction()
    prompt = f"""
        The model predicted that the current Person is <i>{current_prediction}</i>. <br>
        If you have questions about the prediction, select questions from the right and I will answer them.
        """
    instance_dict["initial_prompt"] = prompt

    # get the current prediction
    current_prediction = bot_dict[user_id].get_current_prediction()
    instance_dict["current_prediction"] = current_prediction
    return instance_dict


@bp.route('/get_test_datapoint', methods=['GET'])
def get_test_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    instance_id, instance_dict, prediction_proba = bot_dict[user_id].get_next_instance(train=False)
    instance_dict["id"] = str(instance_id)
    # Make sure all values are strings
    for key, value in instance_dict.items():
        # turn floats to strings if float has zeroes after decimal point
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        instance_dict[key] = str(value)
    return instance_dict


@bp.route("/get_questions", methods=['POST'])
def get_questions():
    """Load the questions."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the questions")
        try:
            response = bot_dict[user_id].get_questions_and_attributes()
        except Exception as ext:
            app.logger.info(f"Traceback getting questions: {traceback.format_exc()}")
            app.logger.info(f"Exception getting questions: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
        return response


@bp.route("/get_response", methods=['POST'])
def get_bot_response():
    """Load the box response."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the bot response")
        try:
            data = json.loads(request.data)
            conversation = bot_dict[user_id].conversation
            question_id = data["question"]
            feature_id = data["feature"]
            response = bot_dict[user_id].update_state_dy_id(question_id, conversation, feature_id)
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
        print(f'MICHI STYLE DEBUG: ${response}')
        return response


app = Flask(__name__)
app.register_blueprint(bp, url_prefix=args.baseurl)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    # clean up storage file on restart
    app.logger.info(f"Launching app from config: {args.config}")
    app.run(debug=False, port=4455, host='0.0.0.0', use_reloader=False)
