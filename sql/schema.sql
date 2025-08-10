-- SQL schema for the synthetic score data

CREATE TABLE scores (
    customer_id TEXT,
    score REAL,
    label INTEGER,
    as_of_date DATE
);
