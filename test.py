import numpy as np
import pandas as pd
import scipy.stats
import sklearn as sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier

from auto_cat.cat_discover import CatDiscover
from auto_cat.auto_cat import AutoCat
from auto_cat.transformers.naive_cat import NaiveCategories
from evaluate.data_prep.australian_weather import AustralianWeather
from evaluate.data_prep.car_prices import CarPrices
from evaluate.data_prep.generate_dataset import GenerateDataset

pd.set_option("display.max_columns", 10000)
pd.set_option("display.precision", 5)
pd.set_option("expand_frame_repr", False)

dataset_models = [
    dict(name='car_prices',
         data=pd.read_csv('./evaluate/datasets/Car_Prices_Poland_Kaggle.csv').sample(10000),
         prep=CarPrices(),
         model=[RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_depth=100),
                LinearRegression(),
                XGBRegressor(max_depth=5, n_estimators=20)],
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error],
         target_method=np.exp),

    dict(name='synthetic_regression',
         data=None,
         prep=GenerateDataset(data_type='regression'),
         model=[
             RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=10),
             LinearRegression(),
             XGBRegressor(max_depth=5, n_estimators=20)],
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error]),

    dict(name='synthetic_linear_regression',
         data=None,
         prep=GenerateDataset(data_type='linear_regression'),
         model=[
             RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=100),
             LinearRegression(),
             XGBRegressor(max_depth=5, n_estimators=20)],
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error]),

    dict(name='synthetic_classification',
         data=None,
         prep=GenerateDataset(data_type='classification', n_classses=2),
         model=[RandomForestClassifier(n_estimators=20, min_samples_leaf=1, max_depth=20),
                LogisticRegression(max_iter=500),
                XGBClassifier(max_depth=20, n_estimators=20)
                ],
         metrics=(sklearn.metrics.accuracy_score,
                  sklearn.metrics.precision_score,
                  sklearn.metrics.recall_score,
                  sklearn.metrics.f1_score
                  )),

    dict(name='australian_weather',
         data=pd.read_csv('./evaluate/datasets/weatherAUS.csv'),
         prep=AustralianWeather(),
         model=[RandomForestClassifier(n_estimators=20, min_samples_leaf=3, max_features=0.5, max_depth=10, class_weight='balanced'),
                LogisticRegression(max_iter=500, class_weight='balanced'),
                XGBClassifier(max_depth=10, n_estimators=10, scale_pos_weight=80)
                ],
         metrics=[sklearn.metrics.accuracy_score, sklearn.metrics.precision_score, sklearn.metrics.recall_score, sklearn.metrics.f1_score]),
]


def experiment_report(name, dataset_params, model, X, Y):
    Y_pred = model.predict(X)
    metric_results = dict(r2=model.score(X, Y), **{metric.__name__: metric(Y, Y_pred) for metric in dataset_params['metrics']})
    return metric_results


def experiment(name, dataset_params, X_train, X_test, Y_train, Y_test):
    model = sklearn.base.clone(dataset_params['model'])
    model.fit(X_train, Y_train)
    metric_results_train = experiment_report(f'{name}-train', dataset_params, model, X_train, Y_train)
    metric_results_test = experiment_report(f'{name}-test', dataset_params, model, X_test, Y_test)
    metric_results = pd.DataFrame([pd.Series(metric_results_train, name='train'), pd.Series(metric_results_test, name='test')]).transpose()
    return metric_results


def experiment_on_dataset(dataset):
    data = dataset['data']
    X, Y = dataset['prep'].transform(data)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    naive_categories = NaiveCategories()
    cat_columns = CatDiscover().discover_categories(X)
    naive_categories.fit_transform(X, cat_columns)
    X_train_naive = naive_categories.transform(X_train)
    X_test_naive = naive_categories.transform(X_test)
    naive_results = experiment('NAIVE', dataset, X_train_naive, X_test_naive, Y_train, Y_test)

    auto_categories = AutoCat()
    Y_train_auto = Y_train if dataset.get('target_method') is None else dataset['target_method'](Y_train)
    X_train_auto = auto_categories.fit_transform(X_train, Y_train_auto)
    X_test_auto = auto_categories.transform(X_test)
    auto_results = experiment('AUTO', dataset, X_train_auto, X_test_auto, Y_train, Y_test)

    df = naive_results.join(auto_results, lsuffix='_naive', rsuffix='_auto')
    df['train_impr'] = df['train_auto'] / df['train_naive']
    df['test_impr'] = df['test_auto'] / df['test_naive']
    return df


def experiment_multiple_times(dataset_of_model, n_times=20):
    dfs = [experiment_on_dataset(dataset_of_model).reset_index() for _ in range(n_times)]
    df_pvalue = pd.concat(dfs).groupby('index').agg(list)
    df_pvalue = df_pvalue['test_impr'].apply(lambda x: scipy.stats.ttest_1samp(x, 1)[1])
    df_aggs = pd.concat(dfs).groupby('index').agg(['mean', 'std'])
    df_aggs.columns = ['_'.join(c) for c in df_aggs.columns.to_flat_index()]
    df_aggs = df_aggs.join(df_pvalue)
    df_aggs['dataset'] = dataset_of_model['name']
    df_aggs['model'] = dataset_of_model['model'].__class__.__name__
    return df_aggs


if __name__ == '__main__':
    dataframes = []
    for dataset in dataset_models:
        for model in dataset['model']:
            dataset_of_model = dataset.copy()
            dataset_of_model['model'] = model
            df = experiment_multiple_times(dataset_of_model)
            print(df)
            dataframes.append(df)

    dataframes = pd.concat(dataframes)

