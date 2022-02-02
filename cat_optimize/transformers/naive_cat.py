import pandas as pd


class NaiveCategories:
    def transform(self, data: pd.DataFrame, columns: list):
        for column in columns:
            data[column] = data[column].astype('category').cat.codes
        return data