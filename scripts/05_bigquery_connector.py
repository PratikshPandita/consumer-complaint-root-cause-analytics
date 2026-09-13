"""
05_bigquery_connector.py
---------------------------------------------------------------------
THE REAL DATA PULL. Run this once you have a free Google Cloud
project set up (see README "Getting a free GCP + BigQuery account").
It queries the genuine public dataset directly -- no credit card
needed to READ public datasets, and BigQuery's free tier covers
1 TB of on-demand queries per month, far more than this project needs.

Usage:
    pip install google-cloud-bigquery pandas db-dtypes
    gcloud auth application-default login      # one-time browser login
    python 05_bigquery_connector.py --project YOUR_GCP_PROJECT_ID

This overwrites data/sample_complaints.csv with REAL records, in the
same schema the rest of the pipeline (03_ and 04_) already expects --
so once this runs, scripts 03 and 04 are analyzing genuine CFPB data
with zero other changes needed.
---------------------------------------------------------------------
"""
import argparse
import pandas as pd

QUERY = """
    SELECT
      complaint_id,
      date_received,
      product,
      issue,
      sub_issue,
      consumer_complaint_narrative,
      company,
      state,
      submitted_via,
      company_response_to_consumer,
      timely_response,
      consumer_disputed
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE consumer_complaint_narrative IS NOT NULL
      AND date_received >= DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)
    ORDER BY RAND()
    LIMIT 20000
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Your GCP project ID")
    parser.add_argument(
        "--out", default="/home/claude/complaints_project/data/sample_complaints.csv"
    )
    args = parser.parse_args()

    from google.cloud import bigquery  # imported here so the script
    # still shows its structure even before the package is installed

    client = bigquery.Client(project=args.project)
    print("Running query against bigquery-public-data.cfpb_complaints ...")
    df = client.query(QUERY).to_dataframe()
    print(f"Pulled {len(df):,} real complaint records.")
    df.to_csv(args.out, index=False)
    print(f"Saved to {args.out} -- scripts 03_ and 04_ will now use real data.")


if __name__ == "__main__":
    main()
