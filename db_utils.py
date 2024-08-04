import yaml
from sqlalchemy import create_engine
import pandas as pd
from typing import Dict

def load_credentials(filepath: str) -> Dict[str, str]:
    with open(filepath, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials

class RDSDatabaseConnector:
    def __init__(self, credentials: Dict[str, str]):
        self._host = credentials['RDS_HOST']
        self._password = credentials['RDS_PASSWORD']
        self._user = credentials['RDS_USER']
        self._database = credentials['RDS_DATABASE']
        self._port = credentials['RDS_PORT']
        self._engine = None

    def _create_engine(self):
        connection_string = f"postgresql://{self._user}:{self._password}@{self._host}:{self._port}/{self._database}"
        self._engine = create_engine(connection_string)

    def fetch_data(self, table_name: str) -> pd.DataFrame:
        if self._engine is None:
            self._create_engine()
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self._engine)

    def save_data_to_csv(self, data: pd.DataFrame, filename: str):
        data.to_csv(filename, index=False)

if __name__ == "__main__":
    credentials = load_credentials('credentials.yaml')
    connector = RDSDatabaseConnector(credentials)
    loan_data = connector.fetch_data('loan_payments')
    connector.save_data_to_csv(loan_data, 'loan_payments.csv')
