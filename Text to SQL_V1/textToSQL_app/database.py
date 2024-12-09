import pandas as pd
import sqlite3
import os

def load_csv_to_sqlite(csv_file, db_file, table_name):
    """
    Load CSV data into an SQLite database. If the database and table already exist,
    skip creating the table and loading the data.
    """
    # Check if the database file exists
    if os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check if the table exists in the database
        cursor.execute(f"""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name='{table_name}';
        """)
        table_exists = cursor.fetchone()

        conn.close()

        if table_exists:
            print(f"Table '{table_name}' already exists in the database '{db_file}'. Skipping creation.")
            return  # Bypass loading data
        else:
            print(f"Table '{table_name}' does not exist. Creating table and loading data...")
    else:
        print(f"Database '{db_file}' does not exist. Creating database and table...")

    # Create the table and load the CSV
    conn = sqlite3.connect(db_file)
    df = pd.read_csv(csv_file, encoding='latin-1')
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data from {csv_file} loaded into SQLite table '{table_name}'.")


def execute_sql_query(db_file, sql_query):
    """
    Execute an SQL query and return the results as a DataFrame.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return pd.DataFrame(result, columns=columns)
    except sqlite3.Error as e:
        conn.close()
        return f"SQL error: {e}"
