# Import necessary libraries

import datetime
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator

# Libraries required for the PythonOperator function
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
from google.cloud import storage

BRONZE_BUCKET = 'dwh_bucket_bronze'       # Your BRONZE GCS bucket name
SILVER_BUCKET = 'dwh_bucket_silver'       # Your SILVER GCS bucket name
GOLD_BUCKET = 'dwh_bucket_gold'         # Your GOLD GCS bucket name

def extract_from_gsheets_and_load_to_bronze():
    """
    Connects to Google Sheets, fetches data into a pandas DataFrame,
    and uploads it as a Parquet file to the GCS Bronze bucket.
    """

        # 1. Authenticate with Google Sheets
    print("Authenticating with Google Sheets...")
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly'
    ]

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)

with DAG(
    dag_id = 'health_insurance_lakehouse_pipeline',
    default_args = default_args,
    description = 'ELT pipeline for health insurance data from GSheets to Databricks',
    schedule_interval = '@daily', # Runs the pipeline once per day
    catchup = False,
) as dag:
    

    pass# --- 4. Define the Tasks ---
    # Task 1: Extract from GSheets and load to Bronze GCS bucket    