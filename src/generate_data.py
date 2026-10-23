
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


NUM_RECORDS = 1000
OUTPUT_FILE = Path("data/raw/test_executions.csv")

random.seed(42)


tests = [
    ("LoginTest", "Verify user login functionality"),
    ("CheckoutTest", "Verify checkout process"),
    ("PaymentTest", "Verify payment processing"),
    ("SearchTest", "Verify product search"),
    ("CartTest", "Verify shopping cart functionality"),
    ("ProfileTest", "Verify user profile update"),
    ("LogoutTest", "Verify user logout"),
    ("RegistrationTest", "Verify new user registration"),
    ("PasswordResetTest", "Verify password reset"),
    ("OrderHistoryTest", "Verify order history"),
]

environments = ["production", "staging", "qa"]

failure_messages = {
    "REAL_BUG": [
        "Assertion failed: expected value does not match actual value",
        "Element displayed incorrect value",
        "Unexpected API response",
        "Incorrect calculation result",
        "Request timed out",
        "Unexpected service response",
    ],
    "FLAKY": [
        "Timeout waiting for element",
        "Element not found after waiting",
        "Test exceeded expected execution time",
        "Intermittent connection timeout",
        "Unexpected API response",
        "Service unavailable",
    ],
    "ENVIRONMENT": [
        "Database connection timeout",
        "Service unavailable",
        "Network connection refused",
        "Environment service did not respond",
        "Request timed out",
        "Unexpected API response",
    ],
    "DATA_ISSUE": [
        "Invalid test data",
        "Required test data not found",
        "Test data validation failed",
        "Duplicate test data detected",
        "Assertion failed: expected value does not match actual value",
        "Unexpected API response",
    ],
}


def generate_execution():
    test_name, test_description = random.choice(tests)

    category = random.choices(
        ["REAL_BUG", "FLAKY", "ENVIRONMENT", "DATA_ISSUE"],
        weights=[25, 35, 25, 15],
        k=1,
    )[0]

    # Failure probability varies, but no category determines status.
    failure_probability = {
        "REAL_BUG": 0.65,
        "FLAKY": 0.45,
        "ENVIRONMENT": 0.55,
        "DATA_ISSUE": 0.50,
    }[category]

    status = (
        "FAIL"
        if random.random() < failure_probability
        else "PASS"
    )

    # Overlapping execution times across categories.
    execution_time = round(random.uniform(3, 45), 2)

    # Retry counts overlap across categories.
    retry_count = random.choices(
        [0, 1, 2, 3],
        weights=[40, 30, 20, 10],
        k=1,
    )[0]

    error_message = (
        random.choice(failure_messages[category])
        if status == "FAIL"
        else ""
    )

    # Any category can occur in any environment.
    environment = random.choice(environments)

    timestamp = datetime(2026, 1, 1) + timedelta(
        minutes=random.randint(0, 60 * 24 * 180)
    )

    return {
        "test_name": test_name,
        "test_description": test_description,
        "execution_time": execution_time,
        "status": status,
        "error_message": error_message,
        "environment": environment,
        "retry_count": retry_count,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "failure_category": category,
    }


def generate_dataset():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_name",
        "test_description",
        "execution_time",
        "status",
        "error_message",
        "environment",
        "retry_count",
        "timestamp",
        "failure_category",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(NUM_RECORDS):
            writer.writerow(generate_execution())

    print(f"Generated {NUM_RECORDS} test executions.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
