
spreadsheet_id=  "1JVqKCiPHg0HM7-5R4N9aHmROJD-YDomA5tCbBvRT2xw/edit?gid=570890881#gid=570890881"

elt_dag = DAG(
    dag_id='elt_pipeline'
)

with DAG(
    dag_id = 'health_insurance_lakehouse_pipeline',
#    default_args = default_args,
    description = 'ELT pipeline for health insurance data from GSheets to Databricks',
    schedule_interval = '@daily', # Runs the pipeline once per day
    catchup = False,
) as elt_dag:

    upload_sheet_to_gcs = GoogleSheetsToGCSOperator(
        task_id="upload_sheet_to_gcs",
        destination_bucket='dwh_bucket_bronze',
        spreadsheet_id= spreadsheet_id,
        range = 'Sleep_Health_and_Lifestyle_Dataset'
        gcp_conn_id="google_cloud_default",
        export_format="CSV",
    )
    

    pass