"""
Pipeline to detect Task and its Urgency
"""

from prompts import TASK_EXTRACTION_PROMPT
import json

def audio_transcription(client, filename):
    """
    Transcribe an audio file using the Groq API.
    """
    try:
        # Open the audio file
        with open(filename, "rb") as file:
            # Create a transcription of the audio file
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),  # Audio file
                model="whisper-large-v3-turbo",  # Model to use for transcription
                response_format="json",  # Response format
                language="en",  # Language of the audio
                temperature=0.0  # Sampling temperature
            )
        
        # Return the transcription response
        return transcription.text

    except Exception as e:
        print(f"Error occurred while transcribing audio file: {e}")
        return None
    
def extract_task(client, audio_transcription):
    """
    Extract task and its urgency from the transcription
    """    
    
    # Define the messages for the conversation
    messages = [
        {
            "role": "system",
            "content": TASK_EXTRACTION_PROMPT
        },
        {
            "role": "user",
            "content": audio_transcription
        }
    ]

    # Create a chat completion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )

    tasks = json.loads(chat_completion.choices[0].message.content)

    # Return the generated content 
    return tasks