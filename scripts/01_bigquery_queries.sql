-- =====================================================================
-- CFPB Complaints Root-Cause & Regulatory Reporting Analytics
-- Target table: `bigquery-public-data.cfpb_complaints.complaint_database`
-- This is a Google-hosted PUBLIC dataset -- no ETL/ingestion needed.
-- Just open BigQuery, create/select a project, and run these directly
-- against the public table. Cost: $0 for the storage (public datasets
-- are hosted by Google); query cost is covered by BigQuery's free
-- 1 TB/month on-demand tier for a personal/free-tier project.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 0. Sanity check / row count + date range
-- ---------------------------------------------------------------------
SELECT
  COUNT(*)                    AS total_complaints,
  MIN(date_received)          AS earliest_date,
  MAX(date_received)          AS latest_date
FROM `bigquery-public-data.cfpb_complaints.complaint_database`;


-- ---------------------------------------------------------------------
-- 1. Complaint volume trend by month and product
--    -> "identifying key trends across geographies" (time dimension)
-- ---------------------------------------------------------------------
SELECT
  DATE_TRUNC(date_received, MONTH)   AS complaint_month,
  product,
  COUNT(*)                            AS complaint_count
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
WHERE date_received >= DATE_SUB(CURRENT_DATE(), INTERVAL 36 MONTH)
GROUP BY complaint_month, product
ORDER BY complaint_month DESC, complaint_count DESC;


-- ---------------------------------------------------------------------
-- 2. Root-cause breakdown: top Issues within each Product
--    -> directly mirrors "identify root cause of complaints"
-- ---------------------------------------------------------------------
WITH ranked_issues AS (
  SELECT
    product,
    issue,
    COUNT(*) AS issue_count,
    RANK() OVER (PARTITION BY product ORDER BY COUNT(*) DESC) AS rnk
  FROM `bigquery-public-data.cfpb_complaints.complaint_database`
  WHERE date_received >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
  GROUP BY product, issue
)
SELECT product, issue, issue_count
FROM ranked_issues
WHERE rnk <= 5
ORDER BY product, rnk;


-- ---------------------------------------------------------------------
-- 3. Geographic trend surfacing (state-level root-cause hot spots)
--    -> "identifying key trends across geographies"
-- ---------------------------------------------------------------------
SELECT
  state,
  issue,
  COUNT(*) AS complaint_count
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
WHERE date_received >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
  AND state IS NOT NULL
GROUP BY state, issue
QUALIFY RANK() OVER (PARTITION BY state ORDER BY COUNT(*) DESC) = 1
ORDER BY complaint_count DESC
LIMIT 20;


-- ---------------------------------------------------------------------
-- 4. Regulatory SLA / timely-response reporting
--    -> "providing accurate and timely updates on complaints to
--       the regulators" + "Support Regulatory, Operational, and
--       ad-hoc reporting"
-- ---------------------------------------------------------------------
SELECT
  company,
  COUNT(*)                                                   AS total_complaints,
  COUNTIF(timely_response = 'Yes')                           AS timely_responses,
  ROUND(COUNTIF(timely_response = 'Yes') / COUNT(*) * 100, 1) AS pct_timely,
  COUNTIF(consumer_disputed = 'Yes')                         AS disputed_count,
  ROUND(COUNTIF(consumer_disputed = 'Yes') / COUNT(*) * 100, 1) AS pct_disputed
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
WHERE date_received >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
GROUP BY company
HAVING total_complaints >= 200
ORDER BY pct_timely ASC, total_complaints DESC
LIMIT 25;


-- ---------------------------------------------------------------------
-- 5. Company response effectiveness
--    -> supports "enabling process improvements across businesses"
-- ---------------------------------------------------------------------
SELECT
  company_response_to_consumer,
  COUNT(*)                                     AS complaint_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
WHERE date_received >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
GROUP BY company_response_to_consumer
ORDER BY complaint_count DESC;


-- ---------------------------------------------------------------------
-- 6. Sample extraction for the root-cause / GenAI validation model
--    Pulls narrative + true label (Issue) for a stratified sample.
--    -> feeds scripts/04_complaint_classifier_validation.py
-- ---------------------------------------------------------------------
SELECT
  complaint_id,
  product,
  issue,
  sub_issue,
  consumer_complaint_narrative,
  state,
  company,
  date_received
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
WHERE consumer_complaint_narrative IS NOT NULL
  AND date_received >= DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)
ORDER BY RAND()
LIMIT 20000;
