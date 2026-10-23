from datetime import datetime

import pandas as pd
import mysql.connector


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        database="ai_test_intelligence",
    )


def import_csv_to_mysql():
    """Import the dataset into MySQL. Run only when the table is empty."""
    df = pd.read_csv("data/raw/test_executions.csv")

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM test_executions")
        count = cursor.fetchone()[0]

        if count > 0:
            print(f"Import skipped: table already contains {count} records.")
            return

        query = """
            INSERT INTO test_executions (
                test_name, execution_time, status, error_message,
                environment, retry_count, failure_category, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        rows = []

        for _, row in df.iterrows():
            error_message = (
                None if pd.isna(row["error_message"])
                else str(row["error_message"])
            )

            rows.append((
                row["test_name"],
                float(row["execution_time"]),
                row["status"],
                error_message,
                row["environment"],
                int(row["retry_count"]),
                row["failure_category"],
                pd.to_datetime(row["timestamp"]).to_pydatetime(),
            ))

        cursor.executemany(query, rows)
        connection.commit()
        print(f"Imported {cursor.rowcount} records into MySQL.")

    finally:
        cursor.close()
        connection.close()


def save_prediction(evidence: dict, prediction: dict) -> int:
    """Save a new test execution and its model prediction."""
    timestamp = datetime.fromisoformat(evidence["timestamp"])

    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO test_executions (
            test_name,
            execution_time,
            status,
            error_message,
            environment,
            retry_count,
            timestamp,
            predicted_category,
            prediction_confidence
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        evidence["test_name"],
        evidence["execution_time"],
        evidence["status"].upper(),
        evidence.get("error_message", ""),
        evidence.get("environment", "qa").lower(),
        evidence.get("retry_count", 0),
        timestamp,
        prediction["predicted_failure_category"],
        prediction.get("confidence"),
    )

    try:
        cursor.execute(query, values)
        connection.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    import_csv_to_mysql()
