import pandas as pd

class DataFrameInfo:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def describe_columns(self):
        print(self.data.dtypes)

    def statistical_summary(self):
        print(self.data.describe(include='all'))

    def distinct_values(self):
        print(self.data.nunique())

    def shape(self):
        print(f"Rows: {self.data.shape[0]}, Columns: {self.data.shape[1]}")

    def null_values_summary(self):
        null_values = self.data.isnull().sum()
        null_percentage = (null_values / self.data.shape[0]) * 100
        null_summary = pd.DataFrame({'count': null_values, 'percentage': null_percentage})
        print(null_summary)
