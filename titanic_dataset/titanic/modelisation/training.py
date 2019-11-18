from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

#fonction de génération de X et Y
def get_X_Y(df, x_features, y_feature):
    print("----------------------------------------------")
    print("Génération de X et Y")
    return df[x_features], df[y_feature]

def model_learning(mod_used, X, y):
    print("----------------------------------------------")
    print("Démarrage de l'apprentissage")
    list_test_size = [a/20.0 for a in list(range(0,20,1))][1:]
    scores = []
    for ts in list_test_size:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=0)
        clf = mod_used.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    return np.array(scores).mean()

def grid_search_params(model1, X, y):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    #max_features = ['auto', 'sqrt']
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
    max_depth.append(None)
    
    grid_search = {
     'n_estimators': n_estimators,
     #'max_features': max_features,
     'min_samples_split': min_samples_split,
     'min_samples_leaf': min_samples_leaf,
     'max_depth': max_depth
     }
    
    gdsr_random = RandomizedSearchCV(estimator = model1, param_distributions = grid_search, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    gdsr_random.fit(X_train, y_train)
    
    print(gdsr_random.best_params_)