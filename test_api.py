import os
import traceback
from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask app ve blueprint tanımlamaları
app = Flask(__name__)
bp = Blueprint('api', __name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@bp.route("/", methods=['GET'])
def home():
    return jsonify({"message": "Speech-to-Text API is running"})

@bp.route("/speech-to-text", methods=['POST'])
def transcribe_audio():
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
        app.logger.error(f"Error transcribing audio: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Error transcribing audio: {str(e)}"}), 500

# Register blueprint
app.register_blueprint(bp)

if __name__ == "__main__":
    print("Starting Speech-to-Text API server...")
    print(f"OpenAI API Key: {'Configured' if openai.api_key else 'NOT CONFIGURED'}")
    app.run(debug=True, port=5000) 