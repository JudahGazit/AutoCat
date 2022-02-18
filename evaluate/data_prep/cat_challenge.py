import numpy as np
import pandas as pd
import scipy.stats

from evaluate.data_prep.data_prep_base import DataPrepBase


class CatChallenge(DataPrepBase):
    def __init__(self):
        self.target_column = 'target'

    def transform(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        data = data.copy()
        data = data.drop(columns=['id'])
        return data.drop(columns=[self.target_column]), data[self.target_column]


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.precision", 5)
    pd.set_option("expand_frame_repr", False)

    data_file = '../datasets/train.csv'
    data = pd.read_csv(data_file)
    print(CatChallenge().transform(data))
