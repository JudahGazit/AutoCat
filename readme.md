# AutoCat: Automated Embedding of Categorical Features

## Usage
    
    from evaluate.data_prep.generate_dataset import GenerateDataset
    from cat_optimize.cat_discover import CatDiscover
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split 

    
    X, Y = GenerateDataset(data_type='regression').transform()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    autocat = CatOptimizer()
    X_train = autocat.fit_transform(X_train, Y_train)
    X_test = autocat.transform(X_test)

    model = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=10)
    model.fit(X_train, Y_train)
    
    print(model.score(X_test, Y_test))
    