from audio_transcription import transcribe_audio

if __name__ == "__main__":
    # Path to the audio file
    filename = "audio_sample.mp4"

    # API key for Groq
    api_key = "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"

    # Call the transcription function
    transcription = transcribe_audio(filename, api_key)

    # Print the transcription result
    if transcription:
        print("Transcription Text:", transcription.text)