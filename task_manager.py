"""
Manages the Task Memory
"""

import sqlite3


def initialize_database(db_name="tasks.db"):
    """
    Initialize the SQLite database. Create the database and table if they do not exist.

    Args:
        db_name (str): The name of the SQLite database file (default: tasks.db).
    """
    try:
        # Connect to the database (create if it doesn't exist)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                urgency TEXT NOT NULL
            )
        """
        )

        conn.commit()
        conn.close()
        print("Database and table initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")


def add_task(task, urgency, db_name="tasks.db"):
    """
    Add a new task with its urgency to the database.

    Args:
        task (str): The task description.
        urgency (str): The urgency level of the task.
        db_name (str): The name of the SQLite database file (default: tasks.db).
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Insert the task into the database
        cursor.execute(
            "INSERT INTO tasks (task, urgency) VALUES (?, ?)", (task, urgency)
        )

        conn.commit()
        conn.close()
        print(f"Task '{task}' with urgency '{urgency}' added successfully.")
    except Exception as e:
        print(f"Error adding task: {e}")


def get_tasks(db_name="tasks.db"):
    """
    Retrieve all tasks from the database.

    Args:
        db_name (str): The name of the SQLite database file (default: tasks.db).

    Returns:
        list of tuples: Each tuple contains (id, task, urgency).
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Query all tasks
        cursor.execute("SELECT * FROM tasks")
        tasks = cursor.fetchall()

        conn.close()
        return tasks
    except Exception as e:
        print(f"Error retrieving tasks: {e}")
        return []
