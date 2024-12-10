import pandas as pd
import sqlite3
import os

# Load CSV into SQLite
def load_csv_to_sqlite(csv_file, db_file, table_name):
    conn = sqlite3.connect(db_file)
    df = pd.read_csv(csv_file, encoding='latin-1')
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data from {csv_file} loaded into SQLite table {table_name}.")


def execute_sql_query(db_file, sql_query):
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
