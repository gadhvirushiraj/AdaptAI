"""
Intervention Pipeline to judge and create health-interventions
"""

from prompts import INTERVENTION_GEN


import base64
import json
from openai import OpenAI
import streamlit as st

# Define the prompt template
PROMPT = """
You are an expert in workplace well-being and productivity. Your role is to analyze the provided inputs and return actionable interventions to improve the user's stress levels, well-being, and productivity. Provide interventions according to the tasks which you can estimate based on screen capture data. For example for typing suggest to streach hands, for math just gaze out of the screen, for web deisgn something to reduce cognitive load and improve creativity, for coding neck excercises and for data entry leg streaches.

INPUT PARAMETERS:
1. **Stress Level**: The user's current stress state (`stressed` or `not stressed`).
2. **Live Timetable**: A structured breakdown of the user's activities, including desk work, meetings, commuting, etc.
3. **Surroundings**: The user's current environment (e.g., cubicle, office, cafeteria).
4. **Screen Capture Data**: A frame-by-frame textual description of the user's last 12 activities on their screen.

OUTPUT FORMAT:
Provide a JSON object with the following structure:
{
    "immediate_action": "Specific short-term action to relieve stress, improve well-being, and boost productivity.",
    "detailed_analysis": "A detailed explanation of how the user's stress level, activity history, surroundings, and screen data were used to determine the recommendation."
}

RULES:
1. Only return the JSON object.
2. Ensure the JSON is valid and does not include any additional text or formatting.
3. The `immediate_action` must be concise, actionable, and tailored to the user's situation.
4. The `detailed_analysis` must explain how the provided inputs influenced the recommendation.
5. Provide interventions according to the tasks which you can estimate based on screen capture data.
6. For each task provide a different intervention.
"""

# Function to call OpenAI API for generating intervention
def process_intervention_with_openai(stress_level, live_timetable, surroundings, screen_capture_data):
    """
    Generate an intervention using OpenAI GPT based on stress level, live timetable,
    surroundings, and screen capture data.
    """
    # Construct the input prompt with user-provided data
    input_text = f"""
    Stress Level: {stress_level}
    Live Timetable:
    {live_timetable}
    Surroundings: {surroundings}
    Screen Capture Data:
    {"\n".join(screen_capture_data)}
    """

    try:
        # OpenAI API call
        client = OpenAI(api_key="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")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": input_text}
            ]
        )

        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        immediate_action = result["immediate_action"]
        detailed_analysis = result["detailed_analysis"]

        return immediate_action, detailed_analysis

    except json.JSONDecodeError as e:
        print("Failed to parse JSON output:", e)
        print("Raw response:", response.choices[0].message.content)
        return None, None
    except Exception as e:
        print("Error occurred:", e)
        return None, None

def intervention_gen(client,stress_level, live_timetable, surrounding,screen_capture_data):
    """
    LLM call to actually judges if we actually require intevention.

    Args:
        client: The API client used to communicate with the LLM.
        stress_status(str): current stress status.
        time-table(str) : An hourly based concise report of activities performed.
        surrounding(str): A description of the environment or surroundings.
        screen_capture_data (list): A list of screen capture descriptions taken during the session.

    Returns:
        Intervention Generated(str)

    """
    print('Stress Level',stress_level)
    print('Live Timetable',live_timetable) 
    print('Surrounding',surrounding) 

    if isinstance(live_timetable, str):
        sanitized_timetable = live_timetable.replace("\n", " | ")
    else:
        sanitized_timetable = str(live_timetable)

    # if isinstance(live_timetable, str):
    #     sanitized_timetable = live_timetable.replace("\n", " | ")
    # else:
    #     sanitized_timetable = str(live_timetable)


    query = INTERVENTION_GEN.format(
        stress_level=stress_level,
        activity_timetable=live_timetable,
        surrounding_type=surrounding,
        screen_capture_data = "\n".join(screen_capture_data[-12:])
    )

    output = client.chat.completions.create(
        messages=[
            {"role": "user", "content": query},
        ],
        model="llama-3.3-70b-versatile",
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    # Generating intervention using openai
    immediate_action,detailed_analysis = process_intervention_with_openai(stress_level, sanitized_timetable, surrounding, screen_capture_data[-12:])

    return output.choices[0].message.content, immediate_action
