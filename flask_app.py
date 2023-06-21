"""The app main."""
import json
import logging
from logging.config import dictConfig
import os
import traceback

from flask import Flask
from flask import render_template, request, Blueprint
from flask_cors import CORS
import gin

from explain.logic import ExplainBot
from explain.sample_prompts_by_action import sample_prompt_for_action


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

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse application level configs
gin.parse_config_file(args.config)

# Setup the explainbot
BOT = ExplainBot()
#BOT.build_exit_survey_table()


@bp.route('/')
def home():
    """Load the explanation interface."""
    app.logger.info("Loaded Login")
    objective = BOT.conversation.describe.get_dataset_objective()
    return render_template("new.html", currentUserId="user", datasetObjective=objective)


@bp.route('/get_datapoint', methods=['GET'])
def get_datapoint():
    """
    Get a new datapoint from the dataset.
    TODO: Check with michi where experiment handling will be.
    """
    # get_next_instance returns tuple with (id, datapoint_dict)
    resp = BOT.get_next_instance()[1]
    return resp


@bp.route('/get_feature_tooltips', methods=['GET'])
def get_feature_tooltips():
    """
    Get feature tooltips from the dataset.
    """
    resp = BOT.feature_tooltip_mapping()
    return resp


@bp.route("/log_feedback", methods=['POST'])
def log_feedback():
    """Logs feedback"""
    feedback = request.data.decode("utf-8")
    app.logger.info(feedback)
    split_feedback = feedback.split(" || ")

    message = f"Feedback formatted improperly. Got: {split_feedback}"
    assert split_feedback[0].startswith("MessageID: "), message
    assert split_feedback[1].startswith("Feedback: "), message
    assert split_feedback[2].startswith("Username: "), message

    message_id = split_feedback[0][len("MessageID: "):]
    feedback_text = split_feedback[1][len("Feedback: "):]
    username = split_feedback[2][len("Username: "):]

    logging_info = {
        "id": message_id,
        "feedback_text": feedback_text,
        "username": username
    }

    BOT.log(logging_info)
    return ""


@bp.route("/sample_prompt", methods=["Post"])
def sample_prompt():
    """Samples a prompt"""
    data = json.loads(request.data)
    action = data["action"]
    username = data["thisUserName"]

    prompt = sample_prompt_for_action(action,
                                      BOT.prompts.filename_to_prompt_id,
                                      BOT.prompts.final_prompt_set,
                                      real_ids=BOT.conversation.get_training_data_ids())

    logging_info = {
        "username": username,
        "requested_action_generation": action,
        "generated_prompt": prompt
    }
    BOT.log(logging_info)

    return prompt


@bp.route("/get_questions", methods=['POST'])
def get_questions():
    """Load the questions."""
    if request.method == "POST":
        app.logger.info("generating the questions")
        try:
            response = BOT.get_questions_and_attributes()
        except Exception as ext:
            app.logger.info(f"Traceback getting questions: {traceback.format_exc()}")
            app.logger.info(f"Exception getting questions: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
        return response


@bp.route("/get_response", methods=['POST'])
def get_bot_response():
    """Load the box response."""
    if request.method == "POST":
        app.logger.info("generating the bot response")
        try:
            data = json.loads(request.data)
            conversation = BOT.conversation
            question_id = data["question"]
            feature_id = data["feature"]
            response = BOT.update_state_dy_id(question_id, conversation, feature_id)
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
    app.run(debug=False, port=4455, host='0.0.0.0')
