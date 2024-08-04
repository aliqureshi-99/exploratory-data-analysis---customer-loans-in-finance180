import pandas as pd

class DataFrameTransform:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def impute_missing_values(self, strategy='median'):
        if strategy == 'median':
            self.data.fillna(self.data.median(), inplace=True)
        elif strategy == 'mean':
            self.data.fillna(self.data.mean(), inplace=True)
        else:
            raise ValueError("Strategy not supported. Use 'median' or 'mean'.")
        return self.data

    def remove_highly_correlated(self, threshold=0.9):
        corr_matrix = self.data.corr().abs()
        upper = corr_matrix.where(pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.data.drop(columns=to_drop, inplace=True)
        return self.data
