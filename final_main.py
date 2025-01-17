import os
import time
import threading
from collections import Counter
import sqlite3
from groq import Groq
from acs_detection import get_img_desp, get_acs
from get_biostats import short_instance_stats, long_instance_stats
from task_extractor import audio_transcription, extract_task

# Lock for thread-safe database access
db_lock = threading.Lock()

def get_client():
    """
    Set up Groq Client

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
        with sqlite3.connect(db_path) as connection:
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
                Desk_Work VARCHAR(255),
                Commuting VARCHAR(255),
                Eating VARCHAR(255),
                In_Meeting VARCHAR(255),
                pNN50 VARCHAR(255),
                hr_interval VARCHAR(255),
                heart_rate VARCHAR(255)
            );
            """
            cursor.execute(create_live_timetable)

            create_ecg_query = """
            CREATE TABLE IF NOT EXISTS hrv_data (
                mean_rr REAL,
                sdnn REAL,
                nn50 INTEGER,
                pnn50 REAL,
                lf_power REAL,
                hf_power REAL,
                lf_hf_ratio REAL,
                heart_rate REAL
            );
            """
            cursor.execute(create_ecg_query)

            create_task_table = """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT
            );
            """
            cursor.execute(create_task_table)

    except Exception as e:
        print(f"An error occurred: {e}")

def fetch_ecg(db_path):
    """Fetch ECG data from the database."""
    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()
            query = """
            SELECT * FROM data_ecg 
            WHERE timestamp_sensor >= (SELECT MAX(timestamp_sensor) FROM data_ecg) - 1200000
            AND timestamp_sensor <= (SELECT MAX(timestamp_sensor) FROM data_ecg);
            """
            cursor.execute(query)
            return cursor.fetchall()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def push_to_table(query, params, db_path):
    """Thread-safe function to push data to the database."""
    with db_lock:
        try:
            with sqlite3.connect(db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(query, params)
                connection.commit()
        except Exception as e:
            print(f"An error occurred while inserting data: {e}")

def vision_pipeline(client, db_path):
    """Thread function to handle vision pipeline."""
    create_table(db_path)
    pre_frame_act = ""
    activity_class_data = []
    last_timetable_push_time = time.time()
    frame_number = 0

    while True:
        frame = f"./frames/frame_{frame_number}.jpg"
        frame_number += 1
        last_capture_time = time.time()

        img_desp = get_img_desp(client, frame, pre_frame_act)
        vision_output = get_acs(client, img_desp)

        activity_class_data.append(vision_output["activity"])

        push_to_table(
            """
            INSERT INTO vision (timestamp, image_desp, activity, activity_class, criticality, surrounding)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                vision_output["timestamp"],
                img_desp,
                vision_output["activity"],
                vision_output["activity_class"],
                vision_output["criticality"],
                vision_output["surrounding"],
            ),
            db_path
        )

        ecg_data = fetch_ecg(db_path)
        json_hrv = short_instance_stats(ecg_data)

        push_to_table(
            """
            INSERT INTO hrv_data (mean_rr, sdnn, nn50, pnn50, lf_power, hf_power, lf_hf_ratio, heart_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                json_hrv['mean_rr'], json_hrv['sdnn'], json_hrv['nn50'], json_hrv['pnn50'],
                json_hrv['lf_power'], json_hrv['hf_power'], json_hrv['lf_hf_ratio'], json_hrv['heart_rate']
            ),
            db_path
        )

        time_diff = time.time() - last_capture_time
        if time_diff < 20:
            time.sleep(20 - time_diff)

        if len(activity_class_data) == 5:
            start_time = time.strftime("%H:%M", time.localtime(last_timetable_push_time))
            long_hrv = long_instance_stats(ecg_data)
            end_time = time.strftime("%H:%M", time.localtime(time.time()))
            time_interval = f"{start_time} - {end_time}"
            push_to_table(
                """
                INSERT INTO timetable (time_interval, Desk_Work, Commuting, Eating, In_Meeting, pNN50, hr_interval, heart_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    time_interval,
                    Counter(activity_class_data).get("Desk_Work", 0),
                    Counter(activity_class_data).get("Commuting", 0),
                    Counter(activity_class_data).get("Eating", 0),
                    Counter(activity_class_data).get("In_Meeting", 0),
                    long_hrv['pNN50'], long_hrv['hr_interval'], long_hrv['heart_rate']
                ),
                db_path
            )
            last_timetable_push_time = time.time()
            activity_class_data = []

def audio_pipeline(client, db_path):
    """Thread function to handle audio transcription and task extraction."""
    while True:
        last_capture_time = time.time()
        current_time = time.strftime("%Y%m%d_%H%M%S")
        audio_file = f"{current_time}.wav"
        transcription = audio_transcription(client, audio_file)
        extracted_tasks = extract_task(client, transcription)

        for task in extracted_tasks:
            push_to_table(
                "INSERT INTO tasks (task) VALUES (?);",
                (task,),
                db_path
            )

        time_diff = time.time() - last_capture_time
        if time_diff < 20:
            time.sleep(20 - time_diff)

def main():
    """Main function to orchestrate multithreading."""
    db_path = "task.db"
    client = get_client()

    vision_thread = threading.Thread(target=vision_pipeline, args=(client, db_path))
    audio_thread = threading.Thread(target=audio_pipeline, args=(client, db_path))

    vision_thread.start()
    audio_thread.start()

    vision_thread.join()
    audio_thread.join()

if __name__ == "__main__":
    main()
