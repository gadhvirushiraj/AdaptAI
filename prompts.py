"""
Prompt Storage
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

IMG_DESCRIPTION_PROMPT = PromptTemplate(
    input_variables=["pre_frame_act"],
    template="""
    You are tasked with describing a scene from an egocentric (first-person) perspective, focusing solely on what is visible in the environment. Follow these guidelines:

    1. **Direct Observation**:
       - Describe only what is directly observable from the given perspective. 
       - Include visible actions and details of the surroundings without speculating beyond the provided visual context.

    2. **Environmental Details**:
       - Note specific aspects such as:
         - Lighting conditions (e.g., bright, dim, artificial, natural light).
         - Time of day if discernible (e.g., morning, afternoon, evening).
         - Location context (e.g., indoors, outdoors, room type, or setting).

    3. **Restrictions**:
       - Avoid assumptions or inferences about unseen objects, actions, or circumstances.
       - Do not include subjective opinions, personal feelings, or unverifiable details.
       - If body parts (e.g., hands) are not visible, do not speculate on their position or activity.

    **Reference**:
    - Previous frame detected activity: "{pre_frame_act}"

    **Example Description**:
    "The scene shows a brightly lit office space with overhead fluorescent lighting. A desk is visible with a laptop, a coffee mug, and some scattered papers. The background features a large window revealing an overcast sky, indicating it might be late afternoon. No body parts are visible in the frame. Previous frame activity detected: 'typing on the keyboard'."
    """,
)

ACS_TASK = """
    You are tasked with analyzing a detailed description of an image captured from an egocentric perspective. 
    Your goals are:
    
    1. **Activity**: Identify what the person appears to be doing in the image. Provide concise and clear descriptions.
    2. **Criticality**: Determine the criticality level based on the following definitions:
        - **Low**: Minimal focus required, such as routine tasks (e.g., washing hands, drinking water).
        - **Mid**: Moderate focus required, such as walking in a variable environment or similar tasks.
        - **High**: High focus required, such as driving, performing demanding tasks, playing sports, or handling life-threatening situations.
    3. **Surrounding**: Describe the environment or context visible in the image. Include notable objects or elements relevant to understanding the scene.

    **Output Requirements**:
    - Output a LIST with | as seperator, do not add any other text just the list
    - Each key must correspond to the specified categories: `activity`, `criticality`, and `surrounding`.
    - Use concise, specific descriptions while ensuring completeness and relevance.
    """


ACS_EXAMPLES = [
    {
        "description": "The person is sitting at a desk with a laptop open, typing on the keyboard. A cup of coffee is nearby, and there are papers scattered around.",
        "activity": "typing on a laptop",
        "criticality": "Mid",
        "surrounding": "office desk with papers and a coffee cup",
    },
    {
        "description": "The person is walking through a corridor while looking down at their phone. The hallway is well-lit with doors on either side.",
        "activity": "walking while using a phone",
        "criticality": "Mid",
        "surrounding": "office hallway with doors on both sides",
    },
    {
        "description": "The individual is standing near a whiteboard, pointing to a diagram while speaking to a group of seated colleagues.",
        "activity": "presenting to colleagues",
        "criticality": "High",
        "surrounding": "meeting room with a whiteboard and seated audience",
    },
    {
        "description": "The person is using a copy machine in a corner of the office, placing papers into the feeder.",
        "activity": "using a copy machine",
        "criticality": "Low",
        "surrounding": "office corner with a copy machine",
    },
    {
        "description": "The individual is in a video conference, wearing headphones, and taking notes on a notepad.",
        "activity": "participating in a video conference",
        "criticality": "High",
        "surrounding": "workspace with headphones, a laptop, and a notepad",
    },
]

ACS_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["description", "activity", "criticality", "surrounding"],
    template="""**Description**:
    {description}
    **Output**:
    [{activity} | {criticality} | {surrounding}]
    """,
)

ACS_PROMPT = FewShotPromptTemplate(
    examples=ACS_EXAMPLES,
    example_prompt=ACS_EXAMPLE_PROMPT,
    prefix=ACS_TASK,
    suffix="Use the examples above to format your response for the new description below:\n\n{img_desp}",
    input_variables=["img_desp"],
)

TASK_EXTRACTION_PROMPT = """
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