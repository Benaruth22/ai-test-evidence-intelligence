
from datetime import datetime
import os

import mysql.connector
import pandas as pd


def get_connection():
    """Connect to MySQL using environment-based configuration."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "ai_test_intelligence"),
    )


def import_csv_to_mysql():
    """Import the dataset into MySQL only when the table is empty."""
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
    """Save a test execution and its model prediction."""
    timestamp = datetime.fromisoformat(evidence["timestamp"])
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO test_executions (
            test_name, execution_time, status, error_message,
            environment, retry_count, timestamp,
            predicted_category, prediction_confidence
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


def get_monitoring_summary() -> dict:
    """Summarize stored predictions for basic model monitoring."""
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT
                COUNT(*) AS total_predictions,
                ROUND(AVG(prediction_confidence), 4) AS average_confidence,
                MIN(timestamp) AS first_prediction,
                MAX(timestamp) AS latest_prediction
            FROM test_executions
            WHERE predicted_category IS NOT NULL
        """)
        overall = cursor.fetchone()

        cursor.execute("""
            SELECT
                predicted_category AS category,
                COUNT(*) AS count,
                ROUND(AVG(prediction_confidence), 4) AS average_confidence
            FROM test_executions
            WHERE predicted_category IS NOT NULL
            GROUP BY predicted_category
            ORDER BY count DESC
        """)
        categories = cursor.fetchall()

        return {
            "total_predictions": overall["total_predictions"],
            "average_confidence": overall["average_confidence"],
            "first_prediction": (
                overall["first_prediction"].isoformat()
                if overall["first_prediction"] else None
            ),
            "latest_prediction": (
                overall["latest_prediction"].isoformat()
                if overall["latest_prediction"] else None
            ),
            "categories": categories,
        }
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    import_csv_to_mysql()
