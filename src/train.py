
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from src.features import create_features


DATA_PATH = Path("data/raw/test_executions.csv")
MODEL_PATH = Path("models/flaky_test_classifier.joblib")

EXPERIMENT_NAME = "AI-Test-Evidence-Intelligence"
REGISTERED_MODEL_NAME = "TestFailureClassifier"

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

CLASS_LABELS = [
    "DATA_ISSUE",
    "ENVIRONMENT",
    "FLAKY",
    "REAL_BUG",
]


def main():
    df = pd.read_csv(DATA_PATH)
    df["status"] = df["status"].astype(str).str.upper()

    # Root-cause classification applies to failed executions only.
    df = df[df["status"] == "FAIL"].copy()
    df = df.dropna(subset=["failure_category"])

    if df.empty:
        raise ValueError("No failed executions with labels were found.")

    df = create_features(df)

    X = df[FEATURE_COLUMNS].astype("float64")
    y = df["failure_category"].astype(str).str.upper()

    if set(y.unique()) != set(CLASS_LABELS):
        raise ValueError(
            f"Expected all four classes. Found: {sorted(y.unique())}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Failed executions used: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Baseline: always predict the most frequent training class.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )
    baseline_report = classification_report(
        y_test,
        baseline_predictions,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    baseline_f1 = baseline_report["macro avg"]["f1-score"]

    print("\n=== BASELINE: DUMMY CLASSIFIER ===")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Macro F1: {baseline_f1:.4f}")

    with mlflow.start_run(run_name="Baseline-DummyClassifier") as run:
        mlflow.log_params({
            "model_type": "DummyClassifier",
            "strategy": "most_frequent",
            "training_scope": "failed_executions_only",
            "test_size": 0.2,
            "random_state": 42,
        })
        mlflow.log_metrics({
            "accuracy": baseline_accuracy,
            "macro_f1_score": baseline_f1,
            "weighted_f1_score": baseline_report[
                "weighted avg"
            ]["f1-score"],
        })
        mlflow.set_tag("model_role", "baseline")
        print(f"Baseline MLflow run ID: {run.info.run_id}")

    # Main model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    with mlflow.start_run(run_name="RandomForest-Comparison") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        report_text = classification_report(
            y_test,
            predictions,
            labels=CLASS_LABELS,
            zero_division=0,
        )
        report = classification_report(
            y_test,
            predictions,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )

        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]

        print("\n=== RANDOM FOREST ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print("\n=== CLASSIFICATION REPORT ===")
        print(report_text)

        print("\n=== COMPARISON ===")
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        print(f"Random Forest accuracy: {accuracy:.4f}")
        print(f"Baseline macro F1: {baseline_f1:.4f}")
        print(f"Random Forest macro F1: {macro_f1:.4f}")

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42,
            "class_weight": "balanced",
            "test_size": 0.2,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": len(FEATURE_COLUMNS),
            "dataset": str(DATA_PATH),
            "training_scope": "failed_executions_only",
        })

        mlflow.log_metrics({
            "accuracy": accuracy,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1_score": macro_f1,
            "weighted_f1_score": weighted_f1,
            "baseline_accuracy": baseline_accuracy,
            "baseline_macro_f1_score": baseline_f1,
            "accuracy_improvement_over_baseline": (
                accuracy - baseline_accuracy
            ),
            "macro_f1_improvement_over_baseline": (
                macro_f1 - baseline_f1
            ),
        })

        mlflow.log_text(
            report_text,
            "classification_report.txt",
        )

        cm = confusion_matrix(
            y_test, predictions, labels=CLASS_LABELS
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_LABELS,
        ).plot(ax=ax, values_format="d")
        ax.set_title("Failed Test Root-Cause Classification")
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        signature = infer_signature(X_test, predictions)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"\nLocal model saved to: {MODEL_PATH}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
