from pathlib import Path

import joblib
import pandas as pd

from src.features import create_features


MODEL_PATH = Path("models/flaky_test_classifier.joblib")

FEATURE_COLUMNS = [
    "execution_time",
    "retry_count",
    "is_timeout",
    "is_connection_error",
    "is_data_error",
    "is_assertion_error",
    "is_element_error",
    "is_service_error",
    "is_production",
    "is_staging",
    "hour",
    "day_of_week",
]


def predict_failure(test_execution):
    """
    Predict the likely root cause of a failed test execution.

    Args:
        test_execution (dict): Test execution details.

    Returns:
        dict: Predicted category and confidence.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python -m src.train' first."
        )

    model = joblib.load(MODEL_PATH)

    # Convert the input dictionary into a DataFrame.
    df = pd.DataFrame([test_execution])

    # Ensure required columns exist.
    required_columns = [
        "execution_time",
        "retry_count",
        "status",
        "error_message",
        "environment",
        "timestamp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required input fields: {missing_columns}"
        )

    # Normalize input values.
    df["status"] = df["status"].astype(str).str.upper()
    df["error_message"] = df["error_message"].fillna("")
    df["environment"] = (
        df["environment"].astype(str).str.lower()
    )

    if df.loc[0, "status"] != "FAIL":
        raise ValueError(
            "Root-cause classification is intended for failed "
            "test executions. The supplied status must be FAIL."
        )

    if pd.isna(df.loc[0, "execution_time"]) or df.loc[
        0, "execution_time"
    ] < 0:
        raise ValueError(
            "execution_time must be a non-negative number."
        )

    if pd.isna(df.loc[0, "retry_count"]) or df.loc[
        0, "retry_count"
    ] < 0:
        raise ValueError(
            "retry_count must be a non-negative number."
        )

    # Create the same engineered features used during training.
    features = create_features(df)

    # Select exactly the features used by the trained model.
    X = features[FEATURE_COLUMNS].astype("float64")

    # Generate the prediction.
    prediction = model.predict(X)[0]

    # Obtain the model's confidence for the predicted category.
    probabilities = model.predict_proba(X)[0]
    class_labels = list(model.classes_)
    predicted_index = class_labels.index(prediction)
    confidence = float(probabilities[predicted_index])

    return {
        "test_name": test_execution.get(
            "test_name", "UnknownTest"
        ),
        "predicted_category": str(prediction),
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    sample_test = {
        "test_name": "CheckoutTest",
        "test_description": "Verify checkout process",
        "execution_time": 35.5,
        "status": "FAIL",
        "error_message": "Timeout waiting for element",
        "environment": "staging",
        "retry_count": 2,
        "timestamp": "2026-10-22 10:30:00",
    }

    try:
        result = predict_failure(sample_test)

        print("\n================================")
        print("AI TEST FAILURE PREDICTION")
        print("================================")
        print(f"Test: {result['test_name']}")
        print(f"Predicted category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("================================")

    except (ValueError, FileNotFoundError) as error:
        print(f"Prediction error: {error}")
