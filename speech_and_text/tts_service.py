import traceback
from flask import Response, jsonify
from openai import OpenAI

# OpenAI client
openai_client = OpenAI()

def generate_audio_from_text(text: str, voice: str = "alloy"):
    """
    Generates audio from the given text using OpenAI's TTS API.
    
    Args:
        text: The text to convert to speech.
        voice: The voice to use for speech generation (e.g., "alloy", "nova").
               Defaults to "alloy".
               
    Returns:
        A dictionary containing either the audio stream (as bytes) or an error message.
    """
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Create a streaming response
        def generate():
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

        # Return the audio stream
        return {"audio": Response(generate(), mimetype="audio/mpeg")}

    except Exception as e:
        error_message = f"Error generating speech: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        return {"error": error_message}
