from ..audio_transcription import transcribe_audio

if __name__ == "__main__":
    # Path to the WAV file for testing
    wav_file = "../audio_sample.wav"  # Adjust the path as necessary for your setup

    # Call the transcription function
    transcription = transcribe_audio(wav_file)

    # Print the transcription result
    print("Transcribed Text:", transcription)
