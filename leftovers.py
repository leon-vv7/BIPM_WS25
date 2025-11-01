def get_data_from_spreadsheet(gsheet_name, ):
        hook = GSheetsHook( gcp_conn_id="google_conn_id", ) 
        spreadsheet = hook.get_values(spreadsheet='name', range='my-range' ) #spreadsheet is list of values from your spreadsheet.
          #add the rest of your code here. 

    
        
        

    get_data_from_gs = PythonOperator( 
        task_id = 'get_data_from_gs',
        python_callable = get_data_from_spreadsheet(link, title) )