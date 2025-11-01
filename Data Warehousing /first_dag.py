# --- 1. Import necessary libraries ---


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

# --- 2. Define Your Project's Variables ---
# ⚠️ UPDATE ALL VALUES IN THIS SECTION ⚠️

# This is the path to your service account key *inside the Composer environment*.
# A common place is to upload it to the 'data/' folder in your Composer bucket.
SERVICE_ACCOUNT_FILE = '/home/airflow/gcs/data/your-keyfile.json'

# Google Sheets variables
GSPREAD_SHEET_NAME = 'Health Insurance Project Data' # The NAME of your Google Sheet
GSPREAD_WORKSHEET_NAME = 'Sheet1'                   # The NAME of the tab you want to read

# GCS Bucket variables
BRONZE_BUCKET = 'your-project-id-bronze'       # Your BRONZE GCS bucket name
SILVER_BUCKET = 'your-project-id-silver'       # Your SILVER GCS bucket name
GOLD_BUCKET = 'your-project-id-gold'         # Your GOLD GCS bucket name
BRONZE_FILE_NAME = 'raw_health_data.parquet'   # The output file name

# Databricks variables
DATABRICKS_CONN_ID = 'databricks_default'      # The name of your Airflow connection to Databricks
BRONZE_TO_SILVER_NOTEBOOK = '/Path/to/your/bronze_to_silver_notebook'
SILVER_TO_GOLD_NOTEBOOK = '/Path/to/your/silver_to_gold_notebook'


# --- 3. Define the Python Function (Task 1) ---
# This function will be executed by the PythonOperator.
# It extracts from GSheets, converts to Parquet, and loads to the Bronze GCS bucket.

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

    # 2. Get data from Google Sheet
    print(f"Opening sheet: {GSPREAD_SHEET_NAME}...")
    sheet = client.open(GSPREAD_SHEET_NAME)
    worksheet = sheet.worksheet(GSPREAD_WORKSHEET_NAME)
    df = get_as_dataframe(worksheet)
    print(f"Successfully loaded {len(df)} rows from sheet.")
    
    # 3. Save DataFrame to a temporary local Parquet file
    local_temp_file = '/tmp/temp_data.parquet'
    df.to_parquet(local_temp_file, index=False)
    print(f"Saved data to temporary file: {local_temp_file}")

    # 4. Authenticate with GCS and upload the file
    print(f"Uploading to GCS bucket: {BRONZE_BUCKET}...")
    storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
    bucket = storage_client.bucket(BRONZE_BUCKET)
    blob = bucket.blob(BRONZE_FILE_NAME)
    
    blob.upload_from_filename(local_temp_file)
    print(f"Successfully uploaded {BRONZE_FILE_NAME} to {BRONZE_BUCKET}.")
    
    # 5. Clean up temporary file
    os.remove(local_temp_file)


# --- 4. Define Databricks Task Payloads (Tasks 2 & 3) ---

# This JSON defines the cluster and notebook for the Bronze-to-Silver job
bronze_to_silver_params = {
    'new_cluster': {
        'spark_version': '13.3.x-scala2.12', # Use a Databricks LTS version
        'node_type_id': 'n2-standard-4',     # A standard GCP node type
        'num_workers': 2
    },
    'notebook_task': {
        'notebook_path': BRONZE_TO_SILVER_NOTEBOOK,
        'base_parameters': {
            'bronze_gcs_path': f'gs://{BRONZE_BUCKET}/{BRONZE_FILE_NAME}',
            'silver_gcs_path': f'gs://{SILVER_BUCKET}/clean_health_data'
        }
    }
}

# This JSON defines the cluster and notebook for the Silver-to-Gold job
silver_to_gold_params = {
    'new_cluster': {
        'spark_version': '13.3.x-scala2.12',
        'node_type_id': 'n2-standard-4',
        'num_workers': 2
    },
    'notebook_task': {
        'notebook_path': SILVER_TO_GOLD_NOTEBOOK,
        'base_parameters': {
            'silver_gcs_path': f'gs://{SILVER_BUCKET}/clean_health_data',
            'gold_gcs_path': f'gs://{GOLD_BUCKET}/aggregated_health_metrics'
        }
    }
}


# --- 5. Define the DAG ---

default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2025, 10, 27), # Start date for the DAG
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=3),
}

with DAG(
    dag_id='health_insurance_lakehouse_pipeline',
    default_args=default_args,
    description='ELT pipeline for health insurance data from GSheets to Databricks',
    schedule_interval='@daily', # Runs the pipeline once per day
    catchup=False,
) as dag:

    # Task 1: Run the Python function to get data
    task_extract_from_gsheets = PythonOperator(
        task_id='extract_from_google_sheets_to_bronze',
        python_callable=extract_from_gsheets_and_load_to_bronze,
    )

'''  # Task 2: Trigger the first Databricks notebook
    task_bronze_to_silver = DatabricksSubmitRunOperator(
        task_id='transform_bronze_to_silver',
        databricks_conn_id=DATABRICKS_CONN_ID,
        json=bronze_to_silver_params,
    )

    # Task 3: Trigger the second Databricks notebook
    task_silver_to_gold = DatabricksSubmitRunOperator(
        task_id='aggregate_silver_to_gold',
        databricks_conn_id=DATABRICKS_CONN_ID,
        json=silver_to_gold_params,
    )
'''
    # --- 6. Set Task Dependencies ---
    # This defines the order: 1 -> 2 -> 3
task_extract_from_gsheets # >> task_bronze_to_silver >> task_silver_to_gold