import pandas as pd


def create_features(df):
    df = df.copy()

    # -----------------------------
    # Status features
    # -----------------------------

    df["is_failed"] = (
        df["status"] == "FAIL"
    ).astype(int)


    # -----------------------------
    # Error message features
    # -----------------------------

    df["error_message"] = (
        df["error_message"]
        .fillna("")
        .str.lower()
    )

    df["is_timeout"] = (
        df["error_message"]
        .str.contains("timeout|timed out", regex=True)
    ).astype(int)

    df["is_connection_error"] = (
        df["error_message"]
        .str.contains("connection|network", regex=True)
    ).astype(int)

    df["is_data_error"] = (
        df["error_message"]
        .str.contains(
            "data|test data|duplicate|validation",
            regex=True,
        )
    ).astype(int)

    df["is_assertion_error"] = (
        df["error_message"]
        .str.contains(
            "assertion|incorrect|expected|actual",
            regex=True,
        )
    ).astype(int)

    df["is_element_error"] = (
        df["error_message"]
        .str.contains(
            "element|displayed|not found",
            regex=True,
        )
    ).astype(int)

    df["is_service_error"] = (
        df["error_message"]
        .str.contains(
            "service|unavailable|respond",
            regex=True,
        )
    ).astype(int)


    # -----------------------------
    # Environment features
    # -----------------------------

    df["is_production"] = (
        df["environment"] == "production"
    ).astype(int)

    df["is_staging"] = (
        df["environment"] == "staging"
    ).astype(int)


    # -----------------------------
    # Timestamp features
    # -----------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )


    return df