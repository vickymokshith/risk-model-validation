-- Basic quality checks for the scores table

-- Count rows with missing scores
SELECT COUNT(*) AS missing_scores FROM scores WHERE score IS NULL;

-- Calculate average score
SELECT AVG(score) AS average_score FROM scores;
