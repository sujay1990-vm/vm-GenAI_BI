import pandas as pd

def create_table_metadata():
    """
    Create metadata for the table.
    """
    metadata = {
        "Column_Name": [
            "Primary_key", "First_Name", "Last_Name", "Age", "Gender",
            "Marital_Status", "Race", "Day", "Shift", "Nurse_name",
            "Clinical_Notes", "Non_clinical_notes"
        ],
        "Column_Type": [
            "INTEGER", "varchar", "varchar", "INTEGER", "varchar",
            "varchar", "varchar", "varchar", "varchar", "varchar",
            "varchar", "varchar"
        ],
        "Description": [
            "Unique identifier for each record",
            "First name of the senior resident",
            "Last name of the senior resident",
            "Age of the senior resident",
            "Gender of the senior resident (Male/Female)",
            "Marital status of the resident (Single/Married/Widowed/Divorced)",
            "Race of the resident",
            "Day of the week the note was recorded",
            "Shift during which the note was recorded (Morning/Night)",
            "Full Name of the nurse who recorded the note",
            "Detailed clinical notes about the resident's incident or medical event/emergency",
            "Non-clinical notes such as personal preferences or complaints"
        ]
    }
    return pd.DataFrame(metadata)

def get_custom_table_schema():
    """
    Generate a custom schema string for the table.
    """
    schema = {
        "Primary_key": "INTEGER (Structured)",
        "First_Name": "TEXT (Structured)",
        "Last_Name": "TEXT (Structured)",
        "Age": "INTEGER (Structured)",
        "Gender": "TEXT (Structured)",
        "Marital_Status": "TEXT (Structured)",
        "Race": "TEXT (Structured)",
        "Day": "TEXT (Structured)",
        "Shift": "TEXT (Structured)",
        "Nurse_name": "TEXT (Structured)",
        "Clinical_Notes": "TEXT (Unstructured)",
        "Non_clinical_notes": "TEXT (Unstructured)"
    }
    return "\n".join([f"{col}: {desc}" for col, desc in schema.items()])
