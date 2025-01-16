"""
Intervention Pipeline to judge and create health-interventions
"""

from prompts import INTERVENTION_GEN


def intervention_gen(client, stress_status, time_table, surrounding):
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

    query = INTERVENTION_GEN.format(
        stress_status=stress_status,
        activity_timetable=time_table,
        surrounding=surrounding,
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
