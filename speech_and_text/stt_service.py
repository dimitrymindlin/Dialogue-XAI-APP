import os
import traceback
import openai

def transcribe_audio_file(audio_file_path: str):
    """
    Transcribes an audio file using OpenAI's Whisper API.

    Args:
        audio_file_path: The path to the audio file to transcribe.

    Returns:
        A dictionary containing either the transcribed text or an error message.
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(audio_file_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        return {"text": transcript.text}
    except Exception as e:
        error_message = f"Error transcribing audio: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        return {"error": error_message}
