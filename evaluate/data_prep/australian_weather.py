import numpy as np
import pandas as pd

from evaluate.data_prep.data_prep_base import DataPrepBase


class AustralianWeather(DataPrepBase):
    def __init__(self):
        self.target_variable = 'RainTomorrow'
        self.na_value = 'NA'
        self.date = 'Date'
        self.yes_no_map = {'Yes': 1, 'No': 0}
        self.yes_no_columns = ['RainToday', self.target_variable]

    def _replace_na_values(self, data):
        data = data.replace(self.na_value, np.nan)
        data = data[data[self.target_variable].notna()]
        for column in data.columns:
            if data.dtypes[column] == float:
                fill_value = data[column].median()
            else:
                fill_value = data[column].value_counts().head(1).index[0]
            data[column] = data[column].fillna(fill_value)
        return data

    def _date_extraction(self, data):
        data[self.date] = pd.to_datetime(data[self.date])
        data[f'{self.date}_month'] = data[self.date].dt.month
        data[f'{self.date}_day'] = data[self.date].dt.day
        data = data.drop(columns=[self.date])
        return data

    def _map_yes_no_values(self, data):
        for column in self.yes_no_columns:
            data[column] = data[column].map(self.yes_no_map.get)
        return data

    def transform(self, data: pd.DataFrame):
        data = self._map_yes_no_values(data)
        data = self._replace_na_values(data)
        data = self._date_extraction(data)
        return data.drop(columns=[self.target_variable]), data[self.target_variable]


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.precision", 5)
    pd.set_option("expand_frame_repr", False)


    data_file = '../datasets/weatherAUS.csv'
    data = pd.read_csv(data_file)
    print(AustralianWeather().transform(data))
