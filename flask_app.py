"""The app main."""
import json
import logging
import os
import traceback

from flask import Flask, render_template
from flask import request, Blueprint
from flask_cors import CORS
import gin
from flask import jsonify

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


@bp.route('/')
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
    return render_template("index.html", currentUserId=user_id, datasetObjective=objective)


@bp.route('/init', methods=['GET'])
def init():
    """Load the explanation interface."""
    user_id = request.args.get("user_id")
    study_group = request.args.get("study_group")
    if user_id is None or "":
        user_id = "TEST"
    if study_group is None or "":
        study_group = "interactive"
    BOT = ExplainBot(study_group)
    bot_dict[user_id] = BOT
    app.logger.info("Loaded Login and created bot")

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
    bot_dict[user_id].dialogue_manager.reset_state()
    bot_dict[user_id].dialogue_manager.print_transitions()
    return result


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


def get_datapoint(user_id, datapoint_type, return_probability=False):
    """
    Get a datapoint from the dataset based on the datapoint type.
    """
    if user_id is None:
        user_id = "TEST"
    current_instance_with_units, instance_counter = bot_dict[user_id].get_next_instance_triple(datapoint_type,
                                                                                               return_probability=return_probability)
    (instance_id, instance_dict, probas, ml_label) = current_instance_with_units
    instance_dict["id"] = str(instance_id)
    if return_probability:
        instance_dict["probabilities"] = probas
    instance_dict["ml_prediction"] = ml_label
    print(instance_dict)
    return instance_dict


@bp.route('/get_train_datapoint', methods=['GET'])
def get_train_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    user_study_group = bot_dict[user_id].get_study_group()
    instance_dict = get_datapoint(user_id, "train")

    if user_study_group == "interactive":
        prompt = f"""
            The model predicts that the current {bot_dict[user_id].instance_type_naming} is <b>{instance_dict["ml_prediction"]}</b>. <br>
            If you have questions about the prediction, select questions from the right and I will answer them.
            """
    else:  # chat
        prompt = f"""
            The model predicts that the current {bot_dict[user_id].instance_type_naming} is <b>{instance_dict["ml_prediction"]}</b>. <br>
            If you have questions about the prediction, type them in the chat and I will answer them.
            """
    instance_dict["initial_prompt"] = prompt

    if user_study_group == "static":
        # Get the explanation report
        static_report = bot_dict[user_id].get_explanation_report()
        static_report["instance_type"] = bot_dict[user_id].instance_type_naming
        instance_dict["static_report"] = static_report
    return instance_dict


@bp.route('/get_test_datapoint', methods=['GET'])
def get_test_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    return get_datapoint(user_id, "test")


@bp.route('/get_final_test_datapoint', methods=['GET'])
def get_final_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    return get_datapoint(user_id, "final_test")


@bp.route('/get_intro_test_datapoint', methods=['GET'])
def get_intro_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    return get_datapoint(user_id, "intro_test")


@bp.route("/set_user_prediction", methods=['POST'])
def set_user_prediction():
    """Set the user prediction."""
    user_id = request.args.get("user_id")
    data = json.loads(request.data)
    user_prediction = data["user_prediction"]
    if user_id is None:
        user_id = "TEST"
    bot = bot_dict[user_id]
    bot.set_user_prediction(user_prediction)
    return "200 OK"


@bp.route("/get_user_correctness", methods=['GET'])
def get_user_correctness():
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    bot = bot_dict[user_id]
    correctness_string = bot.get_user_correctness()
    print(correctness_string)
    response = {"correctness_string": correctness_string}
    return response


@bp.route("/get_questions", methods=['POST'])
def get_questions():
    """Load the questions."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the questions")
        try:
            response = bot_dict[user_id].get_questions_attributes_featureNames()
        except Exception as ext:
            app.logger.info(f"Traceback getting questions: {traceback.format_exc()}")
            app.logger.info(f"Exception getting questions: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
        return response


@bp.route("/get_feature_ranges", methods=['POST'])
def get_feature_ranges():
    """Load the feature ranges."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the feature ranges")
        try:
            response = bot_dict[user_id].get_feature_ranges()
        except Exception as ext:
            app.logger.info(f"Traceback getting feature ranges: {traceback.format_exc()}")
            app.logger.info(f"Exception getting feature ranges: {ext}")
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
            question_id = data["question"]
            feature_id = data["feature"]
            response = bot_dict[user_id].update_state_new(question_id=question_id, feature_id=feature_id)
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
        return response[0]


@bp.route("/get_nl_response", methods=['POST'])
def get_bot_response_from_nl():
    """Load the box response."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the bot response for nl input")
        try:
            data = json.loads(request.data)
            print(data["message"])
            response, question_id, feature_id = bot_dict[user_id].update_state_new(user_input=data["message"])
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
            question_id = None
            feature_id = None
        return jsonify({"response": response, "question_id": question_id, "feature_id": feature_id})


app = Flask(__name__)
app.register_blueprint(bp, url_prefix=args.baseurl)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    # clean up storage file on restart
    app.logger.info(f"Launching app from config: {args.config}")
    app.run(debug=False, port=4455, host='0.0.0.0', use_reloader=False)
