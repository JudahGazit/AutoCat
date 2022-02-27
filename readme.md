# AutoCat: Automated Embedding of Categorical Features

AutoCat automatically detects the categorical features in the dataset, bins smaller categories together, 
and most importantly - picks the right embedding for them based on the correlation to the target variable.

AutoCat significantly improves the performance of both regression and classification models without adding anymore complexity
to them.

Full Report:
http://link_to_report.pdf

## Usage

First, load a dataset and split into train and test.

```python

from evaluate.data_prep.generate_dataset import GenerateDataset
from sklearn.model_selection import train_test_split 


X, Y = GenerateDataset(data_type='regression').transform()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

```

Usually, categorical features are naively embedded. Here we embed each feature using ordianl numbers.
Let's evaluate this embedding with a simple model:

```python

from auto_cat.cat_discover import CatDiscover
from auto_cat.transformers.naive_cat import NaiveCategories
from sklearn.ensemble import RandomForestRegressor

categorical_columns = CatDiscover().discover_categories(X_train)
naive_transformer = NaiveCategories()
X_train_naive = naive_transformer.fit_transform(X_train)
X_test_naive = naive_transformer.transform(X_test)

model = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=10).fit(X_train_naive, Y_train)

print(model.score(X_test_naive, Y_test)) ## R^2 = 0.62519

```

Now, let's use AutoCat and see the difference. Note that model complexity stays exactly the same.

```python

from auto_cat.auto_cat import AutoCat
from sklearn.ensemble import RandomForestRegressor

autocat = AutoCat()
X_train_auto = autocat.fit_transform(X_train, Y_train)
X_test_auto = autocat.transform(X_test)

model = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=10).fit(X_train_auto, Y_train)

print(model.score(X_test_auto, Y_test))  ## R^2 = 0.71735

```
AutoCat boosted the R^2 by 14.7% without adding any additional complexity to the model.