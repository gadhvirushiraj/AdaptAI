"""
Pipeline to detect Action, Criticality and Surrounding
"""

import base64
from prompts import IMG_DESCRIPTION_PROMPT, ACS_PROMPT


def get_img_desp(client, img_path, pre_frame_act):
    """
    VLM calls to describe the POV view in detail.
    """

    try:
        with open(img_path, "rb") as image_file:
            img = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        raise ValueError("Error, couldnt encode image correctly") from e

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
    LLM call to get activity, criticality and surrounding
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

    return output.choices[0].message.content
