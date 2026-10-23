import pandas as pd

from src.features import create_features


def test_create_features():
    data = {
        "test_name": ["LoginTest"],
        "test_description": ["Verify user login functionality"],
        "execution_time": [35.5],
        "status": ["FAIL"],
        "error_message": ["Timeout waiting for element"],
        "environment": ["staging"],
        "retry_count": [2],
        "timestamp": ["2026-03-18 19:24:00"],
        "failure_category": ["FLAKY"],
    }

    df = pd.DataFrame(data)

    result = create_features(df)

    assert result["is_failed"].iloc[0] == 1
    assert result["is_timeout"].iloc[0] == 1
    assert result["is_connection_error"].iloc[0] == 0
    assert result["is_production"].iloc[0] == 0
    assert result["is_staging"].iloc[0] == 1
    assert result["hour"].iloc[0] == 19
    assert result["day_of_week"].iloc[0] == 2