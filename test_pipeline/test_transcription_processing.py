from process_transcription import process_transcription

if __name__ == "__main__":
    # API key for Groq
    api_key = "your_api_key_here"

    # Specify the model to use
    model = "llama-3.3-70b-versatile"

    # Define the messages for the conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain the importance of fast language models."
        }
    ]

    # Call the function to generate chat completion
    response = process_transcription(
        api_key=api_key,
        model=model,
        messages=messages
    )

    # Print the response
    if response:
        print("Chat Completion:", response)
