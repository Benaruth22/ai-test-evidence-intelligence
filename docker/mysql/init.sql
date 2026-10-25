CREATE TABLE IF NOT EXISTS test_executions (
    id INT NOT NULL AUTO_INCREMENT,
    test_name VARCHAR(255) DEFAULT NULL,
    execution_time FLOAT DEFAULT NULL,
    status VARCHAR(20) DEFAULT NULL,
    error_message TEXT,
    environment VARCHAR(50) DEFAULT NULL,
    retry_count INT DEFAULT NULL,
    failure_category VARCHAR(50) DEFAULT NULL,
    timestamp DATETIME DEFAULT NULL,
    predicted_category VARCHAR(50) DEFAULT NULL,
    prediction_confidence FLOAT DEFAULT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
