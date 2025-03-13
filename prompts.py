"""
Prompt Storage
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

IMG_DESCRIPTION_PROMPT = PromptTemplate(
    input_variables=["pre_frame_act"],
    template="""
    You are tasked with describing a scene from an egocentric (first-person) perspective, focusing solely on what is visible in the environment. Follow these guidelines:
    1. **Direct Observation**: Describe only what is directly visible from the given perspective. Include visible actions and details of the surroundings without speculating beyond the provided visual context.
    2. **Environmental Details**: Note specific aspects such as:
         - Lighting conditions (e.g., bright, dim, artificial, natural light).
         - Time of day if discernible (e.g., morning, afternoon, evening).
         - Location context (e.g., indoors, outdoors, room type, or setting).
    3. **Restrictions**:
       - Avoid assumptions or inferences about unseen objects, actions, or circumstances.
       - Do not include subjective opinions, personal feelings, or unverifiable details.
       - If body parts (e.g., hands) are not visible, do not speculate on their position or activity.
    **Previous frame detected activity:**"{pre_frame_act}"
    **Example Description**:
    "The scene shows a brightly lit office space with overhead fluorescent lighting. A desk is visible with a laptop,a coffee mug, and some scattered papers. The background features a large window evident of natural lightning, indicating it might be late afternoon. No body parts are visible in the frame."
    """,
)

ACS_TASK = """
    Analyze egocentric image descriptions and extract actionable insights across three key dimensions: activity, criticality, and surrounding context.
    
    1. **Activity**: Identify the action the person appears to be performing in the image. Provide clear, concise descriptions
    2. **Best Suited Activity Classification** : Choose one from "Desk_Work" (any work-related), "Commuting" (walking), "Eating" (having lunch, coffee break), "In_Meeting" (socializing, physical meeting, presentations), "Other"
    3. **Criticality**: Assign a criticality level based on the following definitions:
        - **Low**: Routine or minimal focus tasks (e.g., drinking water, organizing papers).
        - **Mid**: Tasks requiring moderate focus (e.g., walking in a crowded space, typing).
        - **High**: Demanding tasks requiring significant focus (e.g., driving, playing sports, presenting to an audience).
    4. **Surrounding**: Describe the visible environment or context. Include notable objects or features relevant to understanding the scene.

    **Output Requirements**:
    - Output a LIST with | as seperator, do not add any other text just the list
    - Each key must correspond to the specified categories: `activity`, 'activity_class', `criticality`, and `surrounding`.
    - Use concise, specific descriptions while ensuring completeness and relevance.
    """


ACS_EXAMPLES = [
    {
        "description": "The person is sitting at a desk with a laptop open, typing on the keyboard. A cup of coffee is nearby, and there are papers scattered around.",
        "activity": "typing on a laptop",
        "activity_class": "Desk_Work",
        "criticality": "Mid",
        "surrounding": "office desk with papers and a coffee cup",
    },
    {
        "description": "The person is walking through a corridor while looking down at their phone. The hallway is well-lit with doors on either side.",
        "activity": "walking while using a phone",
        "activity_class": "Commuting",
        "criticality": "Mid",
        "surrounding": "office hallway with doors on both sides",
    },
    {
        "description": "The individual is standing near a whiteboard, pointing to a diagram while speaking to a group of seated colleagues.",
        "activity": "presenting to colleagues",
        "activity_class": "In-Meeting",
        "criticality": "High",
        "surrounding": "meeting room with a whiteboard and seated audience",
    },
    {
        "description": "The person is using a copy machine in office, placing papers into the feeder.",
        "activity": "using a copy machine",
        "activity_class": "Desk_Work",
        "criticality": "Low",
        "surrounding": "office corner with a copy machine",
    },
    {
        "description": "The individual is in a video conference, wearing headphones, and taking notes on a notepad.",
        "activity": "participating in a video conference",
        "activity_class": "Desk_Work",
        "criticality": "High",
        "surrounding": "workspace with headphones, a laptop, and a notepad",
    },
    {
        "description": "The individual is developing a software application on a computer, typing code and testing the application.",
        "activity": "software development",
        "activity_class": "Desk_Work",
        "criticality": "High",
        "surrounding": "workspace with headphones, a laptop, and a notepad",
    },
    {
        "description": "The individual is doing some data entry work on a computer, typing into a spreadsheet.",
        "activity": "Data Entry",
        "activity_class": "Desk_Work",
        "criticality": "High",
        "surrounding": "workspace with headphones, a laptop, and a notepad",
    },
    {
        "description": "The individual is doing multiple tasks, listening to music on headphones, doing some data entry work on a computer, typing into a spreadsheet, and eating.",
        "activity": "Multitasking",
        "activity_class": "Desk_Work",
        "criticality": "High",
        "surrounding": "workspace with headphones, a laptop, and a notepad",
    },
]

ACS_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["description", "activity", "criticality", "surrounding"],
    template="""**Description**:
    {description}
    **Output**:
    [{activity} | {activity_class} | {criticality} | {surrounding}]
    """,
)

ACS_PROMPT = FewShotPromptTemplate(
    examples=ACS_EXAMPLES,
    example_prompt=ACS_EXAMPLE_PROMPT,
    prefix=ACS_TASK,
    suffix="Use the examples above to format your response for the new description below:\n\n{img_desp}",
    input_variables=["img_desp"],
)

INTERVENTION_PROMPT = """
    You are a workplace wellness assistant responsible for evaluating the following inputs and delivering well-structured recommendations that help the user improve his/her
    physiological state helping them in there task performance and productivity.

    1. **stress_level**: Determine the individual’s current stress status (stressed or not stressed). 
    2. **activity_timetable**: valuate the hourly distribution of the individual’s activities, which include:
    - `time`: The time range (e.g., `8-9 PM`).  
    - `desk-work`: Minutes spent on desk work.  
    - `commuting`: Minutes spent commuting.  
    - `eating`: Minutes spent eating.  
    - `in-meeting`: Minutes spent in physical meetings/conversations.  
    3. **surrounding_type**:  Identify the individual’s current environment (e.g., cubicle, meeting room, office)..  
    4. **screen_capture_data**: Summarize the user’s real-time activities based on screen observations (e.g., "Working on a spreadsheet", "Debugging Python code"). These provide insights into multitasking tendencies and workflow patterns.

    ### Example Format:
    **Input:**  
    - stress_level: [stressed or not stressed]  
    - activity_timetable:  
    ```
    | Time       | Desk Work (min)  | Commuting (min) | Eating (min) | In-Meeting (min)  |
    |------------|------------------|-----------------|--------------|-------------------|
    | X-Y PM     | XX               | XX              | XX           | XX                |
    ```
    - surrounding_type: [e.g., cubicle, meeting room]  
    - screen_capture_data: Summary of the observed activities in the frames.

    **Output: Do not changed output list of list formating, strictly** 
    ["Analysis": (A comprehensive evaluation of the individual’s state, identifying patterns such as frequent context-switching, prolonged inactivity, or inefficient task prioritization.)
    "Interventions": (Quick interventions considering the user’s current surroundings and physical state.)]
"""

INTERVENTION_EXAMPLES = [
    {
        "stress_level": "stressed",
        "activity_timetable": """
            Time,Desk Work (min),Commuting (min),Eating (min),In-Meeting (min),Walking (min)
            8-9 AM,60,0,0,0,0
            9-10 AM,45,0,15,0,5
            10-11 AM,40,0,0,20,5
            11-12 PM,55,0,0,5,5
        """,
        "surrounding_type": "cubicle",
        "screen_capture_data": """
            Frame 1: The user is solving math problems related to geometry and algebra.
            Frame 2: The user is re-checking calculations and writing formulas on a whiteboard.
            Frame 3: The user is switching to a browser to look up mathematical concepts and examples.
            Frame 4: The user is reviewing a practice test and marking questions for review.
        """,
        "output": """
        ["Analysis": "You are intensely focused on math-related tasks with minimal breaks, leading to potential mental fatigue. The frequent switching between solving problems and browsing concepts can disrupt flow and add stress.",
        "Interventions": "Take a minute break after a session of problem-solving. During the break, practice 4-7-8 breathing (inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds) to reduce stress and improve focus."]
        """,
    },
    {
        "stress_level": "not stressed",
        "activity_timetable": """
            Time,Desk Work (min),Commuting (min),Eating (min),In-Meeting (min),Walking (min)
            7-8 AM,20,30,10,0,10
            8-9 AM,50,0,10,0,5
            9-10 AM,40,0,0,20,10
            10-11 AM,45,0,0,15,5
        """,
        "surrounding_type": "office",
        "screen_capture_data": """
            Frame 1: The user is typing a passage from a document at high speed.
            Frame 2: The user is reviewing their typed text for errors and correcting them.
            Frame 3: The user is practicing typing speed on a test platform with a timer.
            Frame 4: The user is adjusting the ergonomics of their keyboard and chair.
        """,
        "output": """
        ["Analysis": "Your focus is on improving typing speed with timed tests and text reviews. While you're not stressed, prolonged typing sessions can lead to hand fatigue or strain if posture and breaks are neglected.",
        "Interventions": Every typing test, stretch your fingers, wrists, and shoulders. Perform simple hand exercises like making a fist and releasing it 10 times."]
        """,
    },
    {
        "stress_level": "stressed",
        "activity_timetable": """
            Time,Desk Work (min),Commuting (min),Eating (min),In-Meeting (min),Walking (min)
            1-2 PM,40,0,10,10,5
            2-3 PM,50,0,0,0,10
            3-4 PM,60,0,0,0,0
            4-5 PM,45,0,0,15,5
        """,
        "surrounding_type": "office",
        "screen_capture_data": """
            Frame 1: The user is working on a website mockup using a design tool like Figma or Adobe XD.
            Frame 2: The user is searching for color palettes and UI inspiration online.
            Frame 3: The user is reviewing client feedback on a shared design document.
            Frame 4: The user is creating a layout for a product page with an emphasis on responsive design.
        """,
        "output": """
        ["Analysis": "Your session is focused on creative and client-driven tasks, but the lack of breaks and overemphasis on visual detail can lead to creative fatigue and stress.",
        "Interventions": "Take a minute walk outdoors or around the office to reset your mind. Avoid looking at screens during this time."]
        """,
    },
    {
        "stress_level": "not stressed",
        "activity_timetable": """
            Time,Desk Work (min),Commuting (min),Eating (min),In-Meeting (min),Walking (min)
            3-4 PM,45,0,0,15,5
            4-5 PM,60,0,0,0,5
            5-6 PM,50,0,0,10,10
            6-7 PM,40,0,20,0,10
        """,
        "surrounding_type": "cubicle",
        "screen_capture_data": """
            Frame 1: The user is entering data into a spreadsheet from scanned documents.
            Frame 2: The user is verifying entries against the original source for accuracy.
            Frame 3: The user is using shortcuts to automate repetitive tasks (e.g., Excel formulas).
            Frame 4: The user is checking the formatting and layout of the spreadsheet.
        """,
        "output": """
        ["Analysis": "Your focus on repetitive data entry tasks with verification processes indicates good attention to detail. However, prolonged desk work can cause physical strain and monotony.",
        "Interventions": "After every session of data entry, stand up and stretch your legs for a minute. Perform neck rolls and wrist stretches to relieve physical strain.",]
        """,
    },
]


INTERVENTION_GEN = FewShotPromptTemplate(
    examples=INTERVENTION_EXAMPLES,
    example_prompt=PromptTemplate(
        input_variables=[
            "stress_level",
            "activity_timetable",
            "surrounding_type",
            "screen_capture_data",
        ],
        template="""Input:
            - Stress Level: {stress_level}
            - Activity Timetable:
            {activity_timetable}
            - Surrounding Type: {surrounding_type}
            - A list of screen capture descriptions taken during the session(Used to deduce current activity): {screen_capture_data}
            Output:
            {output}""",
    ),
    prefix=INTERVENTION_PROMPT,
    suffix=""""Use the examples above to format your response for the input below::
            - Stress Level: {stress_level}
            - Activity Timetable:
            {activity_timetable}
            - Surrounding Type: {surrounding_type}
            - A list of screen capture descriptions taken during the session(Used to deduce current activity): {screen_capture_data}
            Output:
            """,
)

PERSONALIZED_LLM_PROMPT = PromptTemplate(
    input_variables=["input_context"],
    template="""
        You are a sophisticated assistant designed to analyze user data on daily activities and physiological metrics to assess stress, fatigue, or balance. Based on your analysis, you adjust your tone dynamically—starting empathetic and intelligent, then becoming more straightforward and subtle as the conversation progresses.
        You will receive the following inputs:
        - Stress Level: Low or Mild or High
        - Activity Durations: Hours spent on activities such as desk-work, commuting, eating, and meetings
        - Time Interval: Specific hourly periods (e.g., 9:00-10:00) to contextualize activity and physiological data.
        Use activity durations and time intervals to assess workload and physical state, determine if they are overworked, under-rested, or have had a heavy workload. Consider the distribution of activities across time intervals to identify patterns of stress or fatigue. Reflect on how their activities suggest balance or potential fatigue throughout the day.     
        Tone Adjustment
        - High Stress/Fatigue: Adopt a highly motivational and encouraging tone. Offer actionable advice such as taking breaks, practicing mindfulness, or pacing themselves.
        - Moderate Stress/Fatigue: Use a moderately motivational tone. Reinforce progress while suggesting simple actions to maintain balance.
        - Low Stress/Relaxed State: Use a subtle, straightforward tone. Keep responses neutral and supportive, acknowledging their balance and steady progress.
        Dynamic Tone Transition As the conversation progresses, gradually shift your tone to become more straightforward, simple, and subtle. Assume that the ongoing interaction helps the person feel calmer and more grounded.
        Tailor every response based on the analysis of stress levels, workload, and physical state. Empathize with their situation and provide feedback or suggestions accordingly. Reflect an understanding of their day and respond in a way that makes them feel supported and motivated.
        Example Input:
        Time,Desk Work (min),Commuting (min),Eating (min), In-Meeting (min), stress_level
        1-2 PM,50,0,0,10,high
        2-3 PM,30,0,5,25,moderate
        3-4 PM,40,0,20,0,low
        4-5 PM,60,0,0,0,moderate
        input_context = {input_context}
    """,
)

SCREEN_CAPTURE_PROMPT = """
    You are tasked with analyzing a detailed description of an image captured from an egocentric perspective.
    Your task is to extract a concise and accurate description of the activity being performed in the scene.
    PARAMETERS:
    1. **Activity Description**: Identify the specific activity visible in the image. Be precise, especially if the activity involves using a laptop or computer, by guessing the exact task (e.g., writing a report, solving a Sudoku puzzle, debugging code). The description should capture the essence of the activity clearly and succinctly.
    OUTPUT FORMAT:
    Provide a concise description of the activity in one or two lines. Do not include any additional text or formatting, just the description.
    RULES:
    1. Only return the activity description as plain text.
    2. Ensure the description is clear, concise, and accurate.
    3. Do not include any additional commentary, metadata, or formatting outside of the description.
"""
