import pandas as pd

from src.database import get_connection


def get_monitoring_data():
    """Load test execution and prediction records from MySQL."""
    connection = get_connection()

    query = """
        SELECT
            id,
            test_name,
            status,
            environment,
            failure_category,
            predicted_category,
            prediction_confidence,
            timestamp
        FROM test_executions
        ORDER BY timestamp DESC
    """

    try:
        df = pd.read_sql(query, connection)
    finally:
        connection.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["prediction_confidence"] = pd.to_numeric(
            df["prediction_confidence"],
            errors="coerce",
        )

    return df


def calculate_monitoring_metrics(df):
    """Calculate basic prediction monitoring metrics."""
    if df.empty:
        return {
            "total_records": 0,
            "predictions_recorded": 0,
            "average_confidence": None,
            "low_confidence_predictions": 0,
            "predicted_categories": {},
        }

    predictions = df.dropna(subset=["predicted_category"])
    confidence = predictions["prediction_confidence"].dropna()

    return {
        "total_records": len(df),
        "predictions_recorded": len(predictions),
        "average_confidence": (
            round(float(confidence.mean()), 4)
            if not confidence.empty
            else None
        ),
        "low_confidence_predictions": int(
            (confidence < 0.60).sum()
        ),
        "predicted_categories": (
            predictions["predicted_category"]
            .value_counts()
            .to_dict()
        ),
    }


if __name__ == "__main__":
    data = get_monitoring_data()
    metrics = calculate_monitoring_metrics(data)

    print("\n=== AI MODEL MONITORING ===")
    print(f"Total test records: {metrics['total_records']}")
    print(
        "Recorded predictions: "
        f"{metrics['predictions_recorded']}"
    )

    average = metrics["average_confidence"]
    print(
        "Average confidence: "
        f"{average:.2%}" if average is not None
        else "Average confidence: No data"
    )

    print(
        "Low-confidence predictions (<60%): "
        f"{metrics['low_confidence_predictions']}"
    )

    print("\nPredicted categories:")
    for category, count in metrics["predicted_categories"].items():
        print(f"  {category}: {count}")
