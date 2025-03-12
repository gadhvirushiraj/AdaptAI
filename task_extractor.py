"""
Pipeline to detect Task and its Urgency
"""

import os
import wave
import json

from prompts import TASK_EXTRACTION_PROMPT


def split_audio_file(input_file, chunk_duration=15):
    """
    Split an audio file into smaller chunks if the file is too large.

    Args:
        input_file (str): Path to the input audio file (WAV format).
        chunk_duration (int): Duration of each chunk in seconds.

    Returns:
        List[str]: List of file paths for the audio chunks.
    """
    chunks = []
    with wave.open(input_file, "rb") as wav:
        frame_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        total_frames = wav.getnframes()

        frames_per_chunk = chunk_duration * frame_rate

        for i in range(0, total_frames, frames_per_chunk):
            wav.setpos(i)
            chunk_data = wav.readframes(frames_per_chunk)

            chunk_file = f"{input_file}_chunk_{i // frames_per_chunk + 1}.wav"
            with wave.open(chunk_file, "wb") as chunk_wav:
                chunk_wav.setnchannels(channels)
                chunk_wav.setsampwidth(sample_width)
                chunk_wav.setframerate(frame_rate)
                chunk_wav.writeframes(chunk_data)

            chunks.append(chunk_file)

    return chunks


def audio_transcription(
    client, filename, max_file_size=10 * 1024 * 1024, chunk_duration=15
):
    """
    Transcribe an audio file using the Groq API. Splits into smaller chunks if too large.

    Args:
        client: The transcription client (e.g., Groq API client).
        filename (str): Path to the input audio file.
        max_file_size (int): Maximum file size in bytes. Defaults to 10 MB.
        chunk_duration (int): Duration of each chunk in seconds for splitting large files.

    Returns:
        str: Transcribed text from the audio file.
    """
    try:
        if os.path.getsize(filename) > max_file_size:
            print(
                f"File {filename} exceeds {max_file_size / (1024 * 1024):.2f} MB. Splitting into chunks..."
            )
            chunks = split_audio_file(filename, chunk_duration)
        else:
            chunks = [filename]

        transcription_text = ""

        for chunk in chunks:
            print(f"Transcribing chunk: {chunk}")
            with open(chunk, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(chunk, file.read()),  # Audio file
                    model="whisper-large-v3-turbo",  # Model to use for transcription
                    response_format="json",  # Response format
                    language="en",  # Language of the audio
                    temperature=0.0,  # Sampling temperature
                )
                transcription_text += transcription.text + " "

            # Optionally, delete temporary chunk files after use
            if chunk != filename:
                os.remove(chunk)

        return transcription_text.strip()

    except Exception as e:
        print(f"Error occurred while transcribing audio file: {e}")
        return None


def extract_task(client, audio_transcription):
    """
    Extract task and its urgency from the transcription
    """

    # Define the messages for the conversation
    messages = [
        {"role": "system", "content": TASK_EXTRACTION_PROMPT},
        {"role": "user", "content": audio_transcription},
    ]

    # Create a chat completion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    content = chat_completion.choices[0].message.content
    if not content:
        print("Error: Response content is empty.")
        return []

    try:
        tasks = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("Response content was:", content)
        return []

    # Return the generated content
    return tasks
