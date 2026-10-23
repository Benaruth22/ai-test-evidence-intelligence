import streamlit as st
import pandas as pd

from src.database import get_connection

st.set_page_config(
    page_title="AI Test Evidence Intelligence",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 AI Test Evidence Intelligence")
st.caption("AI-powered software test failure analysis")

if st.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=10)
def load_data():
    connection = get_connection()

    query = """
        SELECT
            id,
            test_name,
            execution_time,
            status,
            error_message,
            environment,
            retry_count,
            failure_category,
            timestamp,
            predicted_category,
            prediction_confidence
        FROM test_executions
        ORDER BY id DESC
    """

    try:
        # Use a cursor so pandas doesn't receive a raw MySQL
        # DBAPI connection, which caused the earlier warning.
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        return pd.DataFrame(rows)
    finally:
        connection.close()


try:
    df = load_data()

    if df.empty:
        st.warning("No test execution records found.")
        st.stop()

    df["status"] = df["status"].fillna("UNKNOWN").astype(str).str.upper()
    df["environment"] = (
        df["environment"].fillna("unknown").astype(str).str.lower()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["prediction_confidence"] = pd.to_numeric(
        df["prediction_confidence"], errors="coerce"
    )

    predicted = df["predicted_category"].notna()
    failed = df["status"] == "FAIL"

    total_tests = len(df)
    failed_tests = int(failed.sum())
    prediction_count = int(predicted.sum())

    confidence_values = df.loc[
        predicted, "prediction_confidence"
    ].dropna()

    avg_confidence = (
        confidence_values.mean() * 100
        if not confidence_values.empty
        else None
    )

    # Main summary
    st.subheader("📊 Test Execution Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Test Records", f"{total_tests:,}")
    col2.metric("Failed Tests", f"{failed_tests:,}")
    col3.metric("AI Predictions", f"{prediction_count:,}")
    col4.metric(
        "Average Confidence",
        f"{avg_confidence:.1f}%" if avg_confidence is not None else "N/A",
    )

    st.divider()

    # Model monitoring
    st.subheader("🩺 Model Monitoring")

    prediction_data = df.loc[predicted].copy()

    if prediction_data.empty:
        st.info(
            "No predictions have been recorded yet. "
            "Predictions will appear here when saved through the API."
        )
    else:
        confidence_data = prediction_data[
            "prediction_confidence"
        ].dropna()

        low_confidence_count = int((confidence_data < 0.60).sum())

        monitor_col1, monitor_col2, monitor_col3 = st.columns(3)

        monitor_col1.metric(
            "Predictions with Confidence",
            f"{len(confidence_data):,}",
        )
        monitor_col2.metric(
            "Low Confidence (<60%)",
            f"{low_confidence_count:,}",
        )
        monitor_col3.metric(
            "Missing Confidence",
            f"{int(prediction_data['prediction_confidence'].isna().sum()):,}",
        )

        st.markdown("**Prediction confidence distribution**")

        if confidence_data.empty:
            st.info("No confidence values are available to chart.")
        else:
            confidence_chart = (
                confidence_data.mul(100)
                .rename("Confidence (%)")
                .rename_axis("Prediction")
            )
            st.bar_chart(confidence_chart)

        st.markdown("**Predictions by category**")

        category_counts = (
            prediction_data["predicted_category"]
            .value_counts()
            .rename_axis("Category")
            .rename("Count")
        )
        st.bar_chart(category_counts)

        st.caption(
            "Confidence is the model's estimated probability, "
            "not proof that a prediction is correct. Low confidence "
            "is a review signal, not a confirmed error."
        )

    st.divider()

    # Test failure trends
    st.subheader("📈 Test Failure Trends")

    trend_data = df.dropna(subset=["timestamp"]).copy()

    if trend_data.empty:
        st.info("No valid timestamps are available for trend analysis.")
    else:
        trend_data["date"] = trend_data["timestamp"].dt.date

        daily = (
            trend_data.groupby(["date", "status"])
            .size()
            .unstack(fill_value=0)
        )

        for status in ["PASS", "FAIL"]:
            if status not in daily.columns:
                daily[status] = 0

        st.line_chart(daily[["PASS", "FAIL"]].sort_index())

    st.divider()

    # Environment analysis
    st.subheader("🌍 Test Results by Environment")

    environment_data = (
        df.groupby(["environment", "status"])
        .size()
        .unstack(fill_value=0)
    )
    st.bar_chart(environment_data)

    st.divider()

    # Predicted failure categories
    st.subheader("🧠 AI-Predicted Failure Categories")

    if prediction_data.empty:
        st.info("No AI predictions have been recorded yet.")
    else:
        category_counts = prediction_data["predicted_category"].value_counts()
        st.bar_chart(category_counts)

        st.caption(
            "These are model predictions, not independently "
            "confirmed root causes."
        )

    st.divider()

    # Explore records
    st.subheader("🔎 Explore Test Executions")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        statuses = sorted(df["status"].dropna().unique().tolist())
        selected_status = st.selectbox("Test status", ["All"] + statuses)

    with filter_col2:
        environments = sorted(
            df["environment"].dropna().unique().tolist()
        )
        selected_environment = st.selectbox(
            "Environment", ["All"] + environments
        )

    with filter_col3:
        categories = sorted(
            df["predicted_category"].dropna().unique().tolist()
        )
        selected_category = st.selectbox(
            "AI-predicted category", ["All"] + categories
        )

    filtered = df.copy()

    if selected_status != "All":
        filtered = filtered[filtered["status"] == selected_status]

    if selected_environment != "All":
        filtered = filtered[
            filtered["environment"] == selected_environment
        ]

    if selected_category != "All":
        filtered = filtered[
            filtered["predicted_category"] == selected_category
        ]

    display_columns = [
        "id",
        "test_name",
        "status",
        "environment",
        "predicted_category",
        "prediction_confidence",
        "execution_time",
        "retry_count",
        "timestamp",
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Showing {len(filtered):,} of {total_tests:,} records.")

except Exception as error:
    st.error(f"Could not load dashboard data: {error}")
    st.info(
        "Check that MySQL is running and the table columns "
        "match the query in dashboard.py."
    )
