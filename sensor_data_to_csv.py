import sqlite3
import csv

def export_table_to_csv(db_name, table_name, output_file):
    """
    Export a table from SQLite database to a CSV file.

    Args:
        db_name (str): SQLite database file name.
        table_name (str): Name of the table to export.
        output_file (str): Output CSV file name.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        # Fetch all rows from the table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        # Get column names from the table
        column_names = [description[0] for description in cursor.description]
        
        # Write data to CSV
        with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            writer.writerow(column_names)
            # Write the data
            writer.writerows(rows)
        
        print(f"Exported {table_name} to {output_file} successfully.")
    
    except sqlite3.Error as e:
        print(f"Error exporting {table_name}: {e}")
    
    finally:
        conn.close()

# Database and table information
db_name = "sensor_data.db"

# Export ECG data
export_table_to_csv(db_name, "ecg_data", "ecg_data.csv")

# Export IMU data
export_table_to_csv(db_name, "imu_data", "imu_data.csv")
