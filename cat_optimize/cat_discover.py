import pandas as pd


class CatDiscover:
    """
    Simple categorical features discoverer. Detects every column that is either `string` or `categorical`.
    """
    def discover_categories(self, data: pd.DataFrame):
        return data.dtypes[(data.dtypes == 'object') | (data.dtypes == 'category')].index.tolist()