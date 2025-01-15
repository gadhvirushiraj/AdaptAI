from ..process_transcription import process_transcription_with_openai

if __name__ == "__main__":
    # Sample transcribed audio text to test the function
    transcribed_audio_text = """
    During the meeting, please finalize the budget report, reach out to the marketing team for campaign updates,
    and schedule a client call for next week. Also, prepare a summary of the project status by tomorrow.
    """

    # Call the task extraction function
    tasks = process_transcription_with_openai(transcribed_audio_text)

    # Print the extracted tasks
    print("Extracted Tasks:", tasks)
