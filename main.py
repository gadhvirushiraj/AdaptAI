"""
Main python file to orchastrate all pipelines real-time.
"""

import os
import time
from collections import Counter

import cv2
import sqlite3
from groq import Groq
from acs_detection import get_img_desp, get_acs


def get_client():
    """
    SetUp Groq Client

    Returns:
        groq client
    """
    os.environ["GROQ_API_KEY"] = (
        "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"
    )
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    return client


def create_table(db_path):
    """
    Creates necessary tables in the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        cursor.execute("DROP TABLE IF EXISTS vision;")
        cursor.execute("DROP TABLE IF EXISTS timetable;")

        create_image_query = """
        CREATE TABLE vision (
            timestamp VARCHAR(255),
            image_desp VARCHAR(255),
            activity VARCHAR(255),
            activity_class VARCHAR(255),
            criticality VARCHAR(255),
            surrounding VARCHAR(255)
        );
        """
        cursor.execute(create_image_query)

        create_live_timetable = """
        CREATE TABLE timetable(
            time_interval VARCHAR(255),
            Desk-Work VARCHAR(255),
            Commuting VARCHAR(255),
            Eating VARCHAR(255),
            In-Meetin VARCHAR(255)
        );
        """
        cursor.execute(create_live_timetable)

        connection.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def push_vision_db(
    db_path,
    timestamp,
    image_desp,
    activity,
    activity_class,
    criticality,
    surrounding,
):
    """
    Pushes the provided data into the vision table.

    Args:
        db_path (str): Path to the SQLite database file.
        timestamp (str): Timestamp of the data.
        image_descp (str): Description of the captured image.
        activity (str): Detected activity.
        activity_class (str): Class of the activity.
        criticality (str): Criticality level.
        surrounding (str): Description of the surrounding environment.
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        insert_query = """
        INSERT INTO image_data (timestamp, image_desp, activity, activity_class, criticality, surrounding)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        cursor.execute(
            insert_query,
            (
                timestamp,
                image_desp,
                activity,
                activity_class,
                criticality,
                surrounding,
            ),
        )
        connection.commit()
    except Exception as e:
        print(f"An error occurred while inserting data: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def push_to_timetable(db_path, time_interval, activity_class):
    """
    Pushes the activity class counts for the last 60 minutes into the timetable table.

    Args:
        db_path (str): Path to the SQLite database file.
        time_interval (str): Time interval for the data (e.g., "12:00 - 13:00").
        activity_classes (list): List of activity classes for the last 60 minutes.
    """
    activity_counts = Counter(activity_class)
    desk_work_count = activity_counts.get("Desk-Work", 0)
    commuting_count = activity_counts.get("Commuting", 0)
    eating_count = activity_counts.get("Eating", 0)
    in_meeting_count = activity_counts.get("In-Meeting", 0)

    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        insert_query = """
        INSERT INTO timetable (time_interval, Desk-Work, Commuting, Eating, In-Meeting)
        VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(
            insert_query,
            (
                time_interval,
                desk_work_count,
                commuting_count,
                eating_count,
                in_meeting_count,
            ),
        )
        connection.commit()
    except Exception as e:
        print(f"An error occurred while inserting timetable data: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def main():
    """
    Main function to orchestrate image capturing and data processing.
    """
    db_path = "task.db"
    client = get_client()
    create_table(db_path)

    # get/check camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open video capture device.")
        return

    pre_frame_act = ""
    activity_class_data = []
    last_timetable_push_time = time.time()

    while True:
        ret, frame = cap.read()
        last_capture_time = time.time()
        if not ret:
            print("Error: Unable to capture image.")
            break

        img_desp = get_img_desp(client, frame, pre_frame_act)
        vision_output = get_acs(client, img_desp)

        activity_class_data.append(vision_output["activity"])
        push_vision_db(
            db_path,
            vision_output["timestamp"],
            img_desp,
            vision_output["activity"],
            vision_output["activity_class"],
            vision_output["criticality"],
            vision_output["surrounding"],
        )

        # ensures 1min gap between frame capture
        time_diff = time.time() - last_capture_time
        if time_diff < 60:
            time.sleep(60 - time_diff)
        # update time_table every 60 min
        if len(activity_class_data) == 60:
            start_time = time.strftime(
                "%H:%M", time.localtime(last_timetable_push_time)
            )
            end_time = time.strftime("%H:%M", time.localtime(time.time()))
            time_interval = f"{start_time} - {end_time}"
            push_to_timetable(db_path, time_interval, activity_class_data)
            last_timetable_push_time = time.time()
            activity_class_data = []

    cap.release()


if __name__ == "__main__":
    main()
