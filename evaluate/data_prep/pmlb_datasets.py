import numpy as np
import pandas as pd
import scipy.stats
from pmlb import fetch_data

from evaluate.data_prep.data_prep_base import DataPrepBase


class PMLDBDatasets(DataPrepBase):
    def __init__(self):
        self.target_column = 'target'

    def convert_to_letters(self, num, alphabet='abcdefghijklmnopqrstuvwxyz'):
        output = ''
        while num != 0 or len(output) == 0:
            output += alphabet[num % len(alphabet)]
            num = num // len(alphabet)
        return output

    def transform(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        for column in data.columns:
            if data.dtypes[column] == float and data[column].apply(float.is_integer).all():
                data[column] = data[column].astype(int).astype('category')
                categories = data[column].cat.categories
                categories = {category: self.convert_to_letters(category - categories.min()) for category in categories}
                categories = dict(zip(categories.keys(), np.random.permutation(list(categories.values()))))
                data[column] = data[column].map(categories.get)
                data[column] = data[column].astype('category')
        return data.drop(columns=[self.target_column]), data[self.target_column]


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.precision", 5)
    pd.set_option("expand_frame_repr", False)

    data = fetch_data('1203_BNG_pwLinear')
    print(PMLDBDatasets().transform(data))
