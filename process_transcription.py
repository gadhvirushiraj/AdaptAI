import json
from openai import OpenAI

PROMPT = """
You are an expert in task analysis. Your ONLY purpose is to extract actionable tasks from the provided transcribed speech and return them in the specified JSON format.

TASK EXTRACTION PARAMETERS:
1. Identify actionable tasks mentioned in the transcribed text.
2. Ensure each task is clear, concise, and actionable.

OUTPUT FORMAT:
Provide a JSON object with the following structure:
{
    "tasks": [
        "Task 1",
        "Task 2",
        "Task 3"
    ]
}

RULES:
1. Only return the JSON object.
2. Ensure the JSON is valid and does not include any additional text or formatting.
3. Tasks must be derived accurately based on the provided text.
4. Do not include commentary, explanations, or extra text outside the JSON object.
"""

def process_transcription_with_openai(transcribed_audio_text):
    """Process transcribed audio text with OpenAI GPT to extract tasks."""
    client = OpenAI(api_key=str(st.secrets["api_key"]))
    # Send transcribed text along with the prompt to GPT for task extraction
    task_extraction_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": transcribed_audio_text
            }
        ],
    )

    # Parse the JSON response
    try:
        result = json.loads(task_extraction_response.choices[0].message.content)
        tasks = result["tasks"]
    except json.JSONDecodeError as e:
        print("Failed to parse JSON output:", e)
        print("Raw response:", task_extraction_response.choices[0].message.content)
        return None

    return tasks
