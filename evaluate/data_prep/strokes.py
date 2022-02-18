import numpy as np
import pandas as pd

from evaluate.data_prep.data_prep_base import DataPrepBase


class Strokes(DataPrepBase):
    def __init__(self):
        self.target_variable = 'stroke'
        self.na_value = 'N/A'

    def _replace_na_values(self, data):
        data = data.replace(self.na_value, np.nan)
        data = data[data[self.target_variable].notna()].copy()
        for column in data.columns:
            if data.dtypes[column] == float:
                fill_value = data[column].median()
            else:
                fill_value = 'Unknown'
            data[column] = data[column].fillna(fill_value)
        return data

    def transform(self, data: pd.DataFrame):
        data = data.copy()
        data = data.drop(columns=['id'])
        data = self._replace_na_values(data)
        return data.drop(columns=[self.target_variable]), data[self.target_variable]


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.precision", 5)
    pd.set_option("expand_frame_repr", False)


    data_file = '../datasets/healthcare-dataset-stroke-data.csv'
    data = pd.read_csv(data_file)
    print(Strokes().transform(data))
