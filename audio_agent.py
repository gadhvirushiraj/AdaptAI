import os
import re
import json
import smtplib
from groq import Groq
from ics import Calendar, Event
from langchain_groq import ChatGroq
from email.message import EmailMessage
from langchain.schema import SystemMessage, HumanMessage


def extract_meeting_details(transcription, llm):
    """Uses LLM to extract recipient email, meeting subject, time, and location."""
    messages = [
        SystemMessage(
            content="You are an AI that extracts meeting details from transcripts."
        ),
        HumanMessage(
            content=f"""
            Carefully analyze the transcription below and extract key details in JSON format:
            - recipientEmail: List of all recipient emails mentioned.
            - meetingSubject: Subject of the meeting.
            - startTime: Format YYYY-MM-DD HH:MM:SS.
            - endTime: Format YYYY-MM-DD HH:MM:SS.
            - location: Location of the meeting (leave blank if not mentioned).

            If a day of the week is mentioned, calculate the next occurrence.
            Assume a default duration of 1 hour if unspecified.

            Transcript:
            {transcription}
        """
        ),
    ]

    response = llm(messages)
    content = response.content
    start = content.find("{")

    if start != -1:
        open_braces, end = 0, -1
        for i in range(start, len(content)):
            if content[i] == "{":
                open_braces += 1
            elif content[i] == "}":
                open_braces -= 1
                if open_braces == 0:
                    end = i
                    break

        if end != -1:
            json_block = re.sub(r"//.*", "", content[start : end + 1])
            try:
                return json.loads(json_block)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")

    return {
        "recipient_email": "default@example.com",
        "meeting_subject": "Untitled Meeting",
        "start_time": "2025-03-10 10:00:00",
        "end_time": "2025-03-10 11:00:00",
        "location": "Online",
    }


def decide_action(transcription, llm):
    """Determines the necessary action based on transcription."""
    messages = [
        SystemMessage(
            content="You are an AI that decides the appropriate action based on a meeting transcription."
        ),
        HumanMessage(
            content=f"""
            Analyze the transcription:
            - If it mentions scheduling a meeting, respond with 'calendar'.
            - If it mentions sending an email, respond with 'email'.
            - If it summarizes discussions or decisions, respond with 'summarize'.
            - Otherwise, respond with 'none'.
            Transcript:
            {transcription}
        """
        ),
    ]

    response = llm(messages)
    return response.content.strip().lower()


def send_calendar_invite(
    to_email,
    subject,
    start_time,
    end_time,
    location,
    description,
    smtp_user,
    smtp_password,
):
    """Creates and sends a calendar invite via email."""
    cal = Calendar()
    event = Event(
        name=subject,
        begin=start_time,
        end=end_time,
        location=location,
        description=description,
    )
    cal.events.add(event)

    ics_file = "invite.ics"
    with open(ics_file, "w") as f:
        f.writelines(cal)

    msg = EmailMessage()
    msg["Subject"] = f"Meeting Invitation: {subject}"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content(f"Meeting Details:\n{description}\nLocation: {location}")

    with open(ics_file, "rb") as f:
        msg.add_attachment(
            f.read(), maintype="text", subtype="calendar", filename=ics_file
        )

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

    os.remove(ics_file)


def summarize_meeting(transcription, llm):
    """Summarizes a meeting using LLM."""
    messages = [
        SystemMessage(content="You are an AI that summarizes meetings."),
        HumanMessage(content=f"Summarize this meeting:\n{transcription}"),
    ]
    return llm(messages).content.strip()


def generate_followup_email(summary, llm):
    """Generates a professional follow-up email."""
    messages = [
        SystemMessage(
            content="You are an AI that drafts professional follow-up emails."
        ),
        HumanMessage(
            content=f"Draft a follow-up email based on this summary:\n{summary}"
        ),
    ]
    return llm(messages).content.strip()


def send_email(to_email, subject, body, smtp_user, smtp_password):
    """Sends an email using SMTP."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def process_meeting(transcription, api_key, smtp_user, smtp_password):
    """Processes a meeting transcript and performs necessary actions."""
    llm = ChatGroq(api_key=api_key, model_name="llama3-70b-8192")

    print(f"Transcript: {transcription[:200]}...", end="\n")
    print("Extracting meeting details...", end="\n")
    details = extract_meeting_details(transcription, llm)
    print(f"Extracted: {details}", end="\n")

    print("Deciding action...", end="\n")
    action = decide_action(transcription, llm)
    print(f"Decided action: {action}", end="\n")

    if action == "summarize":
        print("Summarizing meeting...", end="\n")
        summary = summarize_meeting(transcription, llm)
        print(f"Summary: {summary}", end="\n")
    elif action == "email":
        print("Generating follow-up email...", end="\n")
        summary = summarize_meeting(transcription, llm)
        email_body = generate_followup_email(summary, llm)
        send_email(
            details["recipientEmail"],
            f"Follow-up: {details['meetingSubject']}",
            email_body,
            smtp_user,
            smtp_password,
        )
    elif action == "calendar":
        print("Sending calendar invite...", end="\n")
        send_calendar_invite(
            details["recipientEmail"],
            details["meetingSubject"],
            details["startTime"],
            details["endTime"],
            details["location"],
            transcription,
            smtp_user,
            smtp_password,
        )
    else:
        print("No action needed.", end="\n")
