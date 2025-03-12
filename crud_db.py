import csv
import sqlite3


def get_live_timetable(db_lock, db_path):
    table = read_from_table(
        db_lock,
        "SELECT time_interval, Desk_Work, Commuting, Eating, In_Meeting FROM timetable",
        db_path,
    )
    header = "time_interval,Desk_Work,Commuting,Eating,In_Meeting\n"
    rows = [",".join(map(str, row)) for row in table]

    return header + "\n".join(rows)


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
                WHERE timestamp_sensor >= (SELECT MAX(timestamp_sensor) FROM ecg_data) - 60000
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


def read_from_table(db_lock, query, db_path):
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


def push_to_table(db_lock, query, params, db_path):
    """Thread-safe function to push data to the database."""
    with db_lock:
        try:
            with sqlite3.connect(db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(query, params)
                connection.commit()
        except Exception as e:
            print(f"An error occurred while inserting data: {e}")


def input_csv(db_lock, db_path):
    """
    Reads data from the 'input.csv' file and inserts it into the 'timetable' table in the database.
    For testing purposes.
    """
    csv_path = "input.csv"
    try:
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
                push_to_table(db_lock, query, params, db_path)

        print(
            "Data from the CSV file has been successfully inserted into the 'timetable' table."
        )

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
