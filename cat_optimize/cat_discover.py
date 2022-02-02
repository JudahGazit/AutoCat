import pandas as pd


class CatDiscover:
    def discover_categories(self, data: pd.DataFrame):
        return data.dtypes[data.dtypes == 'object'].index.tolist()