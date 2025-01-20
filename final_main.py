import os
import time
import threading
from collections import Counter
import sqlite3
from groq import Groq
from acs_detection import get_img_desp, get_acs
from get_biostats import short_instance_stats, long_instance_stats
from task_extractor import audio_transcription, extract_task
from intervent import intervention_gen
import sounddevice as sd
from scipy.io.wavfile import write, read
from win10toast import ToastNotifier
import numpy as np
import os
import cv2
import mss
import csv

# Query device info for the selected device
device_index = 11  # Replace with your device index (e.g., Realtek Microphone Array)
sd.default.device = device_index

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


def input_csv(db_path):
    """
    Reads data from the 'input.csv' file and inserts it into the 'timetable' table in the database.
    """
    csv_path = "input.csv"
    try:
        # Open the CSV file
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            # Validate expected columns in the CSV file
            required_columns = [
                "time_interval",
                "Desk_Work",
                "Commuting",
                "Eating",
                "In_Meeting",
            ]

            for column in required_columns:
                if column not in reader.fieldnames:
                    raise ValueError(f"Missing required column: {column}")

            # Prepare the SQL query for insertion
            query = """
            INSERT INTO timetable (
                time_interval, Desk_Work, Commuting, Eating, In_Meeting, pNN50, heart_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """

            # Iterate through rows in the CSV file and insert them into the database
            for row in reader:
                params = (
                    row["time_interval"],
                    row["Desk_Work"],
                    row["Commuting"],
                    row["Eating"],
                    row["In_Meeting"],
                    "No data",
                    "No data",
                )

                # Push data to the database using the utility function
                push_to_table(query, params, db_path)

        print(
            "Data from the CSV file has been successfully inserted into the 'timetable' table."
        )

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


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
                heart_rate VARCHAR(255)
            );
            """
            cursor.execute(create_live_timetable)

            create_ecg_query = """
            CREATE TABLE IF NOT EXISTS hrv_data (
                mean_rr REAL,
                pnn50 REAL,
                pnn30 REAL,
                pnn20 REAL,
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


def fetch_ecg(db_path, instance):
    """Fetch ECG data from the database."""
    if instance == "short":
        try:
            with sqlite3.connect(db_path) as connection:
                cursor = connection.cursor()
                query = """
                SELECT * FROM ecg_data 
                WHERE timestamp_sensor >= (SELECT MAX(timestamp_sensor) FROM ecg_data) - 10000
                AND timestamp_sensor <= (SELECT MAX(timestamp_sensor) FROM ecg_data);
                """
                cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    elif instance == "long":
        try:
            with sqlite3.connect(db_path) as connection:
                cursor = connection.cursor()
                query = """
                SELECT * FROM ecg_data 
                WHERE timestamp_sensor >= (SELECT MAX(timestamp_sensor) FROM ecg_data) - 180000
                AND timestamp_sensor <= (SELECT MAX(timestamp_sensor) FROM ecg_data);
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


def read_from_table(query, db_path):
    """Thread-safe function to read data from the database."""
    with db_lock:
        try:
            with sqlite3.connect(db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                return results
        except Exception as e:
            print(f"An error occurred while reading data: {e}")
            return None


def get_live_timetable(db_path):
    table = read_from_table(
        "SELECT time_interval, Desk_Work, Commuting, Eating, In_Meeting FROM timetable",
        db_path,
    )
    header = "time_interval,Desk_Work,Commuting,Eating,In_Meeting\n"
    rows = [",".join(map(str, row)) for row in table]

    return header + "\n".join(rows)

def show_intervention_popup(intervention):
    """
    Display a desktop notification using win10toast.
    """
    toaster = ToastNotifier()
    toaster.show_toast(
        "Intervention Prompt",
        intervention,
        duration=10  # Notification duration in seconds
    )


def intervent_pipeline(client, live_timetable, surrounding, stress_level, screen_capture_data):
    if live_timetable is not None:
        intervent = intervention_gen(
            client, stress_level, live_timetable, surrounding, screen_capture_data
        )
        
        # Trigger desktop notification
        show_intervention_popup(intervent)

        # Log to console for debugging
        print(f"Generated Intervention: {intervent}")
    else:
        print("No intervention generated")


def vision_pipeline(client, db_path):
    """Thread function to handle vision pipeline."""

    create_table(db_path)
    pre_frame_act = ""
    activity_class_data = []
    last_timetable_push_time = time.time()
    frame_number = 0
    live_timetable = None

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to open video capture device.")
        return
    pre_frame_act = ""
    activity_class_data = []
    last_timetable_push_time = time.time()
    frame_number = 0
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Capture the screen
    frame_screen_number = 0
    screen_capture_data = []
    frames_screen_dir = "frames_screen"
    os.makedirs(frames_dir, exist_ok=True)

    # Screen recording setup
    screen_capture = mss.mss()
    monitor = screen_capture.monitors[0]  # Capture the entire screen

    while True:
        # Capture a screenshot
        screenshot = screen_capture.grab(monitor)
        frame_screen = np.array(screenshot)
        frame_screen_number += 1
        frame_screen_path = os.path.join(
            frames_screen_dir, f"frame_{frame_screen_number:04d}.jpg"
        )
        cv2.imwrite(frame_screen_path, frame_screen)

        # Process the frame (same as webcam-based pipeline)
        _, buffer = cv2.imencode(".jpg", frame_screen)
        screen_img_desp = get_img_desp(client, buffer, pre_frame_act, is_screen=True)
        screen_capture_data.append(screen_img_desp)

        ret, frame = cap.read()
        _, buffer = cv2.imencode(".jpg", frame)

        frame_number += 1
        frame_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        last_capture_time = time.time()
        if not ret:
            print("Error: Unable to capture image.")
            break
        img_desp = get_img_desp(client, buffer, pre_frame_act)

        vision_output = get_acs(client, img_desp)

        activity_class_data.append(vision_output["activity_class"])

        print("Frame number :", frame_number)
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
            db_path,
        )

        ecg_data = fetch_ecg("sensor_data.db", "short")
        # print('ecg_data', ecg_data)
        json_hrv = short_instance_stats(ecg_data)
        print('pnn50', json_hrv["metrics"]["pnn50"])
        push_to_table(
            """
            INSERT INTO hrv_data (mean_rr, pnn50, pnn30, pnn20, heart_rate)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                json_hrv["metrics"]["mean_rr"],
                json_hrv["metrics"]["pnn50"],
                json_hrv["metrics"]["pnn30"],
                json_hrv["metrics"]["pnn20"],
                json_hrv["metrics"]["hr"],
            ),
            db_path,
        )

        time_diff = time.time() - last_capture_time
        # For individual frames
        if time_diff < 10:
            time.sleep(10 - time_diff)

        # For collective frame processing
        if len(activity_class_data) == 12:
            start_time = time.strftime(
                "%H:%M", time.localtime(last_timetable_push_time)
            )
            ecg_data = fetch_ecg("sensor_data.db", "long")
            long_hrv = long_instance_stats(ecg_data)
            print('pnn50 long', long_hrv["metrics"]["pnn50"])
            end_time = time.strftime("%H:%M", time.localtime(time.time()))
            time_interval = f"{start_time} - {end_time}"
            push_to_table(
                """
                INSERT INTO timetable (time_interval, Desk_Work, Commuting, Eating, In_Meeting, pNN50, heart_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    time_interval,
                    Counter(activity_class_data).get("Desk_Work", 0),
                    Counter(activity_class_data).get("Commuting", 0),
                    Counter(activity_class_data).get("Eating", 0),
                    Counter(activity_class_data).get("In_Meeting", 0),
                    long_hrv["metrics"]["pnn50"],
                    long_hrv["metrics"]["hr"],
                ),
                db_path,
            )
            pnn50 = long_hrv["metrics"]["pnn50"]
            stress_level = (
                "high" if pnn50 < 20 else "moderate" if 20 <= pnn50 < 50 else "low"
            )
            live_timetable = get_live_timetable(db_path)
            # print('timetable', live_timetable)
            intervent_pipeline(
                client,
                live_timetable,
                vision_output["surrounding"],
                stress_level,
                screen_capture_data,
            )
            last_timetable_push_time = time.time()
            activity_class_data = []


class AudioRecorder:
    def __init__(self, samplerate=48000, channels=2, device_index=11):
        self.samplerate = samplerate
        self.channels = channels
        self.device_index = device_index
        self.buffer = np.array([], dtype=np.int16)
        self.lock = threading.Lock()
        self.is_recording = False

    def start_recording(self):
        self.is_recording = True
        threading.Thread(target=self._record_continuously).start()

    def stop_recording(self):
        self.is_recording = False

    def _record_continuously(self):
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=np.int16,
            callback=self._audio_callback,
        ):
            while self.is_recording:
                time.sleep(0.1)

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        with self.lock:
            self.buffer = np.append(self.buffer, indata)

    def get_audio_snapshot(self, use_full_buffer=False, duration=None):
        """
        Get audio data from the buffer.

        Args:
            use_full_buffer (bool): If True, return the entire buffer.
            duration (int): Number of seconds to retrieve (ignored if use_full_buffer is True).

        Returns:
            np.ndarray: Audio data.
        """
        with self.lock:
            if use_full_buffer:
                return self.buffer
            elif duration:
                num_samples = int(duration * self.samplerate * self.channels)
                return (
                    self.buffer[-num_samples:]
                    if len(self.buffer) >= num_samples
                    else self.buffer
                )
            else:
                return np.array([], dtype=np.int16)


def audio_pipeline(client, db_path, recorder, duration=60):
    """
    Function to process accumulated audio every 15 seconds.

    Args:
        client: The client object for transcription and task extraction.
        db_path: Path to the SQLite database.
        recorder: An instance of `AudioRecorder`.
        duration: Duration in seconds for processing intervals.
    """
    audio_file = "accumulated_audio.wav"

    while True:
        # Wait for the next 15-second interval
        time.sleep(duration)

        # Extract the last `duration` seconds of audio and save to a file
        audio_data = recorder.get_audio_snapshot(duration)
        if len(audio_data) == 0:
            print("No audio data available for processing.")
            continue

        write(audio_file, recorder.samplerate, audio_data)
        print(f"Saved last {duration} seconds of audio to {audio_file}.")

        # Process the audio file
        transcription = audio_transcription(client, audio_file)
        if not transcription:
            print("No transcription generated. Skipping.")
            continue

        extracted_tasks = extract_task(client, transcription)
        if extracted_tasks:
            for task in extracted_tasks:
                push_to_table("INSERT INTO tasks (task) VALUES (?);", (task,), db_path)
            print(f"Extracted tasks: {extracted_tasks}")
        else:
            print("No tasks extracted.")

        # Store extracted tasks in the database
        for task in extracted_tasks:
            push_to_table("INSERT INTO tasks (task) VALUES (?);", (task,), db_path)

        print(f"Extracted tasks: {extracted_tasks}")


def main():
    """Main function to orchestrate multithreading."""
    db_path = "task.db"
    client = get_client()

    # Initialize the audio recorder
    recorder = AudioRecorder()
    # Start continuous audio recording
    recorder.start_recording()

    vision_thread = threading.Thread(target=vision_pipeline, args=(client, db_path))
    #audio_thread = threading.Thread(
    #    target=audio_pipeline, args=(client, db_path, recorder)
    #)

    try:
        vision_thread.start()
        # audio_thread.start()

        # Wait for both threads to finish
        vision_thread.join()
        # audio_thread.join()
    except KeyboardInterrupt:
        print("Stopping processes...")
        recorder.stop_recording()
        vision_thread.join()
        # audio_thread.join()
        print("All processes stopped gracefully.")


if __name__ == "__main__":
    main()
