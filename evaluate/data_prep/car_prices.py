import numpy as np
import pandas as pd
import scipy.stats

from evaluate.data_prep.data_prep_base import DataPrepBase


class CarPrices(DataPrepBase):
    def __init__(self):
        self.price_column = 'price'

    def drop_outliers(self, data):
        log_price = np.log(data[self.price_column])
        iqr = scipy.stats.iqr(log_price)
        data = data[(log_price > log_price.median() - 1.5 * iqr) &
                    (log_price < log_price.median() + 1.5 * iqr)]
        return data


    def transform(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        data = data.drop(columns=data.columns[:1])
        data = data[data['province'] != '(']
        data = self.drop_outliers(data)
        return data.drop(columns=[self.price_column]), np.log(data[self.price_column])


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.precision", 5)
    pd.set_option("expand_frame_repr", False)

    data_file = '../datasets/Car_Prices_Poland_Kaggle.csv'
    data = pd.read_csv(data_file)
    print(CarPrices().transform(data))
