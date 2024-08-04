import pandas as pd

class DataTransform:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def remove_symbols(self, columns: list, symbol: str):
        for column in columns:
            if self.data[column].dtype == 'object':
                self.data[column] = self.data[column].str.replace(symbol, '')
            else:
                print(f"Skipping {column}, not a string column")

    def to_datetime(self, columns: list):
        for column in columns:
            self.data[column] = pd.to_datetime(self.data[column])

    def transform(self) -> pd.DataFrame:
        self.remove_symbols(['loan_amount', 'int_rate'], '%')
        self.to_datetime(['issue_date', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date'])
        return self.data
