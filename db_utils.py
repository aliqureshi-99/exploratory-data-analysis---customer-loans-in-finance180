import yaml
from sqlalchemy import create_engine
import pandas as pd

def load_credentials(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials

class RDSDatabaseConnector:
    def __init__(self, credentials: dict):
        self.host = credentials['RDS_HOST']
        self.password = credentials['RDS_PASSWORD']
        self.user = credentials['RDS_USER']
        self.database = credentials['RDS_DATABASE']
        self.port = credentials['RDS_PORT']

    def init_engine(self):
        connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self.engine = create_engine(connection_string)

    def fetch_data(self, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self.engine)

    def save_data_to_csv(self, dataframe: pd.DataFrame, filename: str):
        dataframe.to_csv(filename, index=False)
