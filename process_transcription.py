from groq import Groq

def process_transcription(api_key, model, messages, temperature=0.5, max_tokens=1024, top_p=1, stop=None, stream=False):
    """
    Generate a chat completion using the Groq API.

    Args:
        - api_key (str): API key for authentication.
        - model (str): The language model to use for the chat completion (e.g., "llama-3.3-70b-versatile").
        - messages (list): List of messages in the format [{"role": "user/system", "content": "message"}].
        - temperature (float): Controls randomness (default: 0.5).
        - max_tokens (int): Maximum number of tokens to generate (default: 1024).
        - top_p (float): Controls diversity via nucleus sampling (default: 1).
        - stop (list or str): Stop sequence(s) to signal the model to stop generating text (default: None).
        - stream (bool): Whether to stream partial responses (default: False).

    Returns:
        - str: Generated completion content.
    """
    try:
        # Initialize the Groq client
        client = Groq(api_key=api_key)

        # Create a chat completion
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=stream,
        )

        # Return the generated content 
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Error occurred while generating chat completion: {e}")
        return None
