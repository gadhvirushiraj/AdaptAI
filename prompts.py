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
       - If you see that the person is doing something, describe the task in detail. If there are a number of tasks describe them in detail.
       - If you see that the person is doing something, describe the task in detail. If there are a number of tasks describe them in detail.

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
    
    1. **Activity**: Identify what the person appears to be doing in the image. Provide concise and clear descriptions. (If on laptop provide an accurate guess of the exact task that the person is doing)
    2. **Best Suited Activity Classification** : Choose one from "Desk_Work" (any work related), "Commuting" (walking), "Eating" (having lunch, coffee break), In_Meeting (having conversation, physical meeting, presentations)
    3. **Criticality**: Determine the criticality level based on the following definitions:
        - **Low**: Minimal focus required, such as routine tasks (e.g., washing hands, drinking water).
        - **Mid**: Moderate focus required, such as walking in a variable environment or similar tasks.
        - **High**: High focus required, such as driving, performing demanding tasks, playing sports, or handling life-threatening situations.
    4. **Surrounding**: Describe the environment or context visible in the image. Include notable objects or elements relevant to understanding the scene.

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

INTERVENTION_PROMPT = """
You are a workplace wellness assistant designed to improve an employee's mental and physical well-being. Analyze the following inputs:

1. **stress_level**: Current stress state of the person (`stressed` or `not stressed`).  
2. **activity_timetable**: Hourly breakdown of the individual's activities, including:
   - `time`: The time range (e.g., `8-9 PM`).  
   - `desk-work`: Minutes spent on desk work.  
   - `commuting`: Minutes spent commuting.  
   - `eating`: Minutes spent eating.  
   - `in-meeting`: Minutes spent in physical meetings/conversations.  
3. **surrounding_type**: The person's current environment (e.g., `meeting room`, `cubicle`, `office`, `cafeteria`).  
4. **screen_capture_data**: Descriptions of activities captured in frames (e.g., "Working on a spreadsheet", "Browsing emails", "Debugging Python code"). These frames represent the person's real-time task context and multitasking tendencies.

#### Output:  
Provide:
- **Analysis**: A detailed assessment of the person’s current state, incorporating patterns from the `activity_timetable` and multitasking behavior observed in the `screen_capture_data`. Highlight potential issues, such as frequent context-switching, prolonged inactivity, or lack of task prioritization.
- **Interventions**: Feasible recommendations to improve well-being and task performance, considering:
   - Stress level
   - Activity history
   - Observed multitasking or focus patterns from the screen capture data.
   - Current surroundings.

Additional Context:
The tasks the person is working on are:

1. **Math Test**: Problem-solving requiring focus and analytical thinking.
2. **Typing Test**: Requires speed and accuracy in typing.
3. **Website Design Task**: Demands creativity, attention to detail, and interaction with design tools.
4. **Data Entry Task**: Involves repetitive tasks that may lead to fatigue or monotony.


#### Guidelines:
Guidelines:
1. **Task-Based Interventions**: Recommend interventions specific to the type of task:
    - **Math Test**: Suggest relaxation and focus strategies to improve analytical thinking and reduce cognitive overload.
    - **Typing Test**: Suggest short hand exercises or breathing techniques to improve typing performance and reduce tension.
    - **Website Design Task**: Recommend short creativity-boosting exercises, such as taking a walk or looking at inspiring designs.
    - **Data Entry Task**: Suggest physical movement or brief mindfulness practices to counter monotony and reduce fatigue.
2. **Activity Insights**: If the `screen_capture_data` indicates frequent multitasking or context-switching, recommend steps to minimize distractions and prioritize single-tasking.
3. **Time-Conscious Suggestions**: Interventions must be actionable and fit within workplace norms, ideally taking 5-15 minutes.
4. **Environment-Aware**: Tailor interventions to the current surroundings, such as suggesting physical activity in open spaces or short mindfulness exercises in quieter areas.
5. **Stress Reduction**: When `stress_level` is `stressed`, focus on relaxation, structured task prioritization, and manageable work blocks. For `not stressed`, focus on maintaining productivity and balance.

### Example Format:
**Input:**  
- stress_level: [stressed or not stressed]  
- activity_timetable:  

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
[
 "Analysis": (Brief summary of patterns/issues)
 "Task Improvement": (Specific steps to enhance productivity and well-being.)
 "Interventions": [ Immediate Action: (Recommendation based on the current location.), Follow-Up: (Actionable steps to improve well-being post-task.) ] 
]
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
        "Interventions": [
            Immediate Action: "Take a minute break after a session of problem-solving. During the break, practice 4-7-8 breathing (inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds) to reduce stress and improve focus.",
            Task Improvement: [
                "Group similar math problems together to avoid context-switching. For example, solve all algebra questions first, then geometry.",
                "Write down formulas and important concepts on a cheat sheet for quick reference to minimize browser distractions."
            ],
            "Follow-Up": [
                "Plan your practice sessions with alternating blocks of intense focus (50 minutes) and short breaks (10 minutes). Use the Pomodoro technique to maintain a steady workflow.",
                "Review incorrect questions in the last 15 minutes of each session to consolidate learning and reduce anxiety before tests."
            ]
        ]
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
        "Interventions": [
            Immediate Action: "Every typing test, stretch your fingers, wrists, and shoulders. Perform simple hand exercises like making a fist and releasing it 10 times.",
            Task Improvement: [
                "Use online tools that provide detailed feedback on typing speed and accuracy. Focus on improving the accuracy first, then work on speed.",
                "Adjust your keyboard position so that your wrists remain straight, and ensure your chair provides good back support."
            ],
            "Follow-Up": [
                "Set daily goals for typing speed improvement (e.g., increase by 5 WPM). Track progress weekly to stay motivated.",
                "Integrate short mindfulness exercises (e.g., deep breathing for 2 minutes) after every 3 sessions to stay relaxed and maintain focus."
            ]
        ]
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
        "Interventions": [
            Immediate Action: "Take a minute walk outdoors or around the office to reset your mind. Avoid looking at screens during this time.",
            Task Improvement: [
                "Create a design checklist before starting the mockup to reduce back-and-forth revisions. For example, define colors, fonts, and layout components upfront.",
                "Use pre-made UI kits or templates to speed up the design process for repetitive elements."
            ],
            "Follow-Up": [
                "Schedule a 15-minute feedback review session at the end of the day to consolidate client input. This minimizes interruptions during design time.",
                "Incorporate intentional breaks for inspiration, such as browsing a curated design gallery like Dribbble or Behance for 5 minutes."
            ]
        ]
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
        "Interventions": [
            Immediate Action: "After every session of data entry, stand up and stretch your legs for a minute. Perform neck rolls and wrist stretches to relieve physical strain.",
            Task Improvement: [
                "Use keyboard shortcuts to speed up repetitive tasks (e.g., Ctrl+D for copying cells). Learn Excel automation techniques like macros or data validation.",
                "Set up a dual-monitor workspace if possible, with one screen for source data and the other for the spreadsheet. This reduces back-and-forth toggling and improves efficiency."
            ],
            "Follow-Up": [
                "Review completed entries in batches instead of one-by-one to save time. This allows for more focused error-checking sessions.",
                "Incorporate small rewards, such as a 5-minute game or snack break after completing each section, to maintain motivation."
            ]
        ]
        """,
    }


]



INTERVENTION_GEN = FewShotPromptTemplate(
    examples=INTERVENTION_EXAMPLES,
    example_prompt=PromptTemplate(
        input_variables=["stress_level", "activity_timetable", "surrounding_type","screen_capture_data"],
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
    input_variables=["pre_frame_act"],
    template="""
        You are an empathetic and intelligent assistant designed to analyze user input data related to their daily activities and physiological metrics. Based on the analysis, you will decide whether the person is stressed, fatigued, or in a balanced state and adjust your tone dynamically. Over time, as the conversation progresses, your tone should become more straightforward, simple, and subtle.

        ### Instructions:

        1. **Input Data:**
        You will be given the following inputs:
        - `HRV Metrics (pNN50)`: A measure of heart rate variability where lower values indicate higher stress, and higher values suggest relaxation.
        - `Activity Durations`: Number of hours spent on specific activity classes like desk work, commuting, eating, and meetings.
        - `HR-interval`: Heart rate interval, which may indicate strain or fatigue when irregular or high.
        - `Time Interval`: This shows the specific hour of the day the data belongs to, in a 24-hour format. The time intervals will be represented as ranges, such as 9:00-10:00, 10:00-11:00, and so on. This allows you to understand the distribution of activities and physiological metrics across different hours of the day.

        2. **Determine State:**
        - Use the HRV (`pNN50`) and HR-interval values to assess the person’s stress level:
            - High Stress: Low pNN50 values combined with high or irregular HR intervals.
            - Moderate Stress: Mid-range pNN50 values with stable HR intervals.
            - Relaxed State: High pNN50 values and steady HR intervals.
        - Use the activity durations and time intervals to assess their workload and physical state:
            - Determine if they are overworked, under-rested, or have had a heavy workload.
            - Consider the distribution of activities across time intervals to identify patterns of stress or fatigue.
            - Reflect on how their activities suggest balance or potential fatigue throughout the day.

        3. **Tone Adjustment:**
        - **High Stress/Fatigue:** Adopt a highly motivational and encouraging tone. Offer actionable advice like taking breaks, practicing mindfulness, or pacing themselves.
        - **Moderate Stress/Fatigue:** Use a moderately motivational tone. Reinforce their progress while suggesting simple actions to maintain balance.
        - **Low Stress/Relaxed State:** Use a subtle, straightforward tone. Keep responses neutral and supportive, acknowledging their balance and steady progress.

        4. **Dynamic Tone Transition:**
        - As the conversation progresses, gradually shift your tone to become more straightforward, simple, and subtle. Assume that the ongoing interaction helps the person feel calmer and more grounded.

        5. **Personalized Responses:**
        - Tailor every response based on the analysis of stress levels, workload, and physical state.
        - Empathize with their situation and provide feedback or suggestions accordingly.
        - Reflect an understanding of how their day has been and respond in a way that makes them feel supported and motivated.
        - The main goal is to provide personalized, tailored output for this person based on their data. The information provided is specific to the individual, and your responses should always reflect that.
        - For example, if the user asks for recommendations—be it for food, a restaurant, or a juice—give suggestions that are best suited for them at that moment, considering their stress levels, energy needs, and overall day’s workload.

        ### Example Input:
        ```json

        Time,Desk Work (min),Commuting (min),Eating (min),In-Meeting (min),Walking (min), pnn50, hr
        1-2 PM,50,0,0,0,10,20, 62
        2-3 PM,30,0,0,30,5,10, 70
        3-4 PM,40,0,0,20,5, 30, 85
        4-5 PM,60,0,0,0,0, 25, 78


        input  = {input}
    """,
)

PERSONALIZED_LLM_PROMPT = PromptTemplate(
    input_variables=["input_context"],
    template="""
        You are an empathetic and intelligent assistant designed to analyze user input data related to their daily activities and physiological metrics. Based on the analysis, you will decide whether the person is stressed, fatigued, or in a balanced state and adjust your tone dynamically. Over time, as the conversation progresses, your tone should become more straightforward, simple, and subtle.

        ### Instructions:

        1. **Input Data:**
        You will be given the following inputs:
        - `HRV Metrics (pNN50)`: A measure of heart rate variability where lower values indicate higher stress, and higher values suggest relaxation.
        - `Activity Durations`: Number of hours spent on specific activity classes like desk work, commuting, eating, and meetings.
        - `HR-interval`: Heart rate interval, which may indicate strain or fatigue when irregular or high.
        - `Time Interval`: This shows the specific hour of the day the data belongs to, in a 24-hour format. The time intervals will be represented as ranges, such as 9:00-10:00, 10:00-11:00, and so on. This allows you to understand the distribution of activities and physiological metrics across different hours of the day.

        2. **Determine State:**
        - Use the HRV (`pNN50`) and HR-interval values to assess the person’s stress level:
            - High Stress: Low pNN50 values combined with high or irregular HR intervals.
            - Moderate Stress: Mid-range pNN50 values with stable HR intervals.
            - Relaxed State: High pNN50 values and steady HR intervals.
        - Use the activity durations and time intervals to assess their workload and physical state:
            - Determine if they are overworked, under-rested, or have had a heavy workload.
            - Consider the distribution of activities across time intervals to identify patterns of stress or fatigue.
            - Reflect on how their activities suggest balance or potential fatigue throughout the day.

        3. **Tone Adjustment:**
        - **High Stress/Fatigue:** Adopt a highly motivational and encouraging tone. Offer actionable advice like taking breaks, practicing mindfulness, or pacing themselves.
        - **Moderate Stress/Fatigue:** Use a moderately motivational tone. Reinforce their progress while suggesting simple actions to maintain balance.
        - **Low Stress/Relaxed State:** Use a subtle, straightforward tone. Keep responses neutral and supportive, acknowledging their balance and steady progress.

        4. **Dynamic Tone Transition:**
        - As the conversation progresses, gradually shift your tone to become more straightforward, simple, and subtle. Assume that the ongoing interaction helps the person feel calmer and more grounded.

        5. **Personalized Responses:**
        - Tailor every response based on the analysis of stress levels, workload, and physical state.
        - Empathize with their situation and provide feedback or suggestions accordingly.
        - Reflect an understanding of how their day has been and respond in a way that makes them feel supported and motivated.
        - The main goal is to provide personalized, tailored output for this person based on their data. The information provided is specific to the individual, and your responses should always reflect that.
        - For example, if the user asks for recommendations—be it for food, a restaurant, or a juice—give suggestions that are best suited for them at that moment, considering their stress levels, energy needs, and overall day’s workload.

        ### Example Input:
        ```json

        Time,Desk Work (min),Commuting (min),Eating (min),In-Meeting (min),Walking (min), pnn50, hr
        1-2 PM,50,0,0,0,10,20, 62
        2-3 PM,30,0,0,30,5,10, 70
        3-4 PM,40,0,0,20,5, 30, 85
        4-5 PM,60,0,0,0,0, 25, 78


        input_context = {input_context}
    """,
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
