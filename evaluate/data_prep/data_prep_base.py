import abc

import pandas as pd


class DataPrepBase(abc.ABC):
    def transform(self, data: pd.DataFrame):
        raise NotImplementedError