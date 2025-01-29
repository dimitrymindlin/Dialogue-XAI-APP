# Readme for the Dialog Xai App Experiments

This repository contains the code for the Dialog Xai App Experiments backend. The experiments are based on the Dialog Xai App, which is a web-based application that allows users to interact with a dialog system and get explanations for an ML models predictions. The experiments are designed to evaluate the effectiveness of different explanation strategies in improving user's model understanding.

## Setup

To run the experiments, you will need to set up the Dialog Xai App and the experiment code. Follow the instructions below to get started.

### Dialog Xai App

1. Install requirements: ```pip install -r requirements.txt```
2. Start backend: ```python -m gunicorn --timeout 0 -b 0.0.0.0:5000 flask_app:app```
3. Start frontend in dialogue_xai_frontend repo.

## Project structure

The project is organized as follows:
flask_app.py: The main Flask application that serves the Dialog Xai App. Starts the backend server and has the api endpoints.
### Main endpoints are:
- /init: Initializes the dialog system and returns the initial state.
- /get_{stage}_datapoint: Returns the next stage data point, one of [train, test, final, initial]
- /get_bot_response: Returns the bot response for the given user input. Similarly, /get_bot_response_from_nl get response from natural language input.

### Main Components are
- `ExplainBot (explain/logic.py)`: The main class that initializes all explanations and orchestrates the conversation.
  - update_state_new (update_state_from_nl) is the main conversation loop that interprets the user input and returns the bot response, accessing precomputed XAI explanations.
- `explain/explanations` has the classes to generate XAI explanations for the model predictions.
- `data` folder has datasets and model training scripts.
- `global_config` defines which config file to use in configs folder. Configs folder has example configs. Current working dataset, model and config are `adult`.
- `llm_agents/mape_k_approach/mape_k_workflow_agent.py` is the current main focus. This agent makes use of MAPE-K model to answer user questions while monitoring and scaffolding the user's understanding of the model. 