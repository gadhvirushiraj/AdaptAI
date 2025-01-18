"""
Intervention Pipeline to judge and create health-interventions
"""

from prompts import INTERVENTION_GEN


def intervention_gen(client,stress_level, live_timetable, surrounding):
    """
    LLM call to actually judges if we actually require intevention.

    Args:
        client: The API client used to communicate with the LLM.
        stress_status(str): current stress status.
        time-table(str) : An hourly based concise report of activities performed.
        surrounding(str): A description of the environment or surroundings.

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
<<<<<<< HEAD
        activity_timetable=sanitized_timetable,
=======
        activity_timetable=live_timetable,
>>>>>>> f2e0a6b29c3749b7eded55108d0471deca1d9a9d
        surrounding_type=surrounding,
    )

    output = client.chat.completions.create(
        messages=[
            {"role": "user", "content": query},
        ],
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return output.choices[0].message.content
