from task_manager import initialize_database, add_task, get_tasks

if __name__ == "__main__":
    # Initialize the database (will create it if it doesn't exist)
    initialize_database()

    # Add tasks to the database
    add_task("Finish project report", "High")
    add_task("Clean the kitchen", "Medium")
    add_task("Plan weekend trip", "Low")

    # Retrieve and print all tasks
    tasks = get_tasks()
    print("\nCurrent Tasks in Database:")
    for task in tasks:
        print(f"ID: {task[0]}, Task: {task[1]}, Urgency: {task[2]}")
