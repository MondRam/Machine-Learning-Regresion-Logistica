CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    modelo VARCHAR(50),
    r2 FLOAT,
    mse FLOAT,
    rmse FLOAT,
    mae FLOAT
);

CREATE TABLE new_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    data JSONB,
    balance REAL
);