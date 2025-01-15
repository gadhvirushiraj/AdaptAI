from groq import Groq

def transcribe_audio(filename, api_key, model="whisper-large-v3-turbo", language="en", temperature=0.0, prompt=None):
    """
    Transcribe an audio file using the Groq API.

    Args:
        - filename (str): Path to the audio file to be transcribed.
        - api_key (str): Groq API key for authentication.
        - model (str): Model to use for transcription (default: whisper-large-v3-turbo).
        - language (str): Language of the audio (default: "en").
        - temperature (float): Sampling temperature for the transcription model (default: 0.0).
        - prompt (str): Optional context or spelling prompt to guide the transcription.

    Returns:
        - dict: The transcription response in JSON format.
    """
    try:
        # Initialize the Groq client
        client = Groq(api_key=api_key)
        
        # Open the audio file
        with open(filename, "rb") as file:
            # Create a transcription of the audio file
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),  # Audio file
                model=model,  # Model to use for transcription
                prompt=prompt,  # Optional prompt
                response_format="json",  # Response format
                language=language,  # Language of the audio
                temperature=temperature  # Sampling temperature
            )
        
        # Return the transcription response
        return transcription

    except Exception as e:
        print(f"Error occurred while transcribing audio file: {e}")
        return None
