"""
Pipeline to detect Action, Criticality, and Surrounding (ACS). Making Live-Timetable.
"""

from datetime import datetime
import base64
from prompts import IMG_DESCRIPTION_PROMPT, ACS_PROMPT


def get_img_desp(client, img, pre_frame_act):
    """
    VLM (Vision-Language Model) calls to describe the POV (point-of-view) view in detail.

    Args:
        client: The API client used to communicate with the VLM.
        img_path (str): The path to the image file.
        pre_frame_act (str): Description of the activity from the previous frame.

    Returns:
        str: A detailed description of the image, combining the POV information
             and pre-frame context.
    """
    if isinstance(img, str):
        try:
            with open(img, "rb") as image_file:
                img = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise ValueError("Error: couldn't encode the image correctly") from e

    query = IMG_DESCRIPTION_PROMPT.format(pre_frame_act=pre_frame_act)
    output = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                        },
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return output.choices[0].message.content


def get_acs(client, img_desp):
    """
    LLM (Large Language Model) call to extract activity, criticality, and surrounding.

    Args:
        client: The API client used to communicate with the LLM.
        img_desp (str): The detailed description of the image generated by the VLM.

    Returns:
        dict: A dictionary with the keys:
            - "activity" (str): The detected activity.
            - "criticality" (str): The criticality level of the scene.
            - "surrounding" (str): A description of the environment or surroundings.
    """

    query = ACS_PROMPT.format(img_desp=img_desp)
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

    try:
        parts = output.choices[0].message.content.strip("[]").split(" | ")
        result = {
            "timestamp": datetime.now(),
            "activity": parts[0].strip(),
            "activity_class": parts[1].strip(),
            "criticality": parts[2].strip(),
            "surrounding": parts[3].strip(),
        }
    except IndexError as e:
        raise ValueError(
            "Response format error: expected '[activity | activity_class | criticality | surrounding]'."
        ) from e

    return result
