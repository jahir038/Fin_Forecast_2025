# import pandas as pd
# def load_and_preprocess_data(file_path):
#     df = pd.read_excel(file_path)
#     return df

import pandas as pd
from sqlalchemy import create_engine
def load_and_preprocess_data_sql(server, database, table_name):
    # Connection string (Windows Authentication)
    conn_str = f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    # Create engine
    engine = create_engine(conn_str)
    # Load full table
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    return df
