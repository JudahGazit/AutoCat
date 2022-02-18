import numpy as np
import pandas as pd
import scipy.stats

from evaluate.data_prep.data_prep_base import DataPrepBase


class Insurance(DataPrepBase):
    def __init__(self):
        self.price_column = 'charges'

    def transform(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        return data.drop(columns=[self.price_column]), np.log(data[self.price_column])


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.precision", 5)
    pd.set_option("expand_frame_repr", False)

    data_file = '../datasets/insurance.csv'
    data = pd.read_csv(data_file)
    print(Insurance().transform(data))
