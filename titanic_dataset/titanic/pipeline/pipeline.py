import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import sys
sys.path.append('titanic/')
import data_preprocessing.preprocessing as pre
import modelisation.model as modl
import modelisation.training as turn

def pipeline_p(file, separator, features_convert, type_convert, features_used_X, feature_used_y, features_to_dumnify, md):
    df = pd.read_csv(file, sep=separator)
    features_tabl_nan = pre.features_nan_detect(df, df.shape[0])
    pre.features_nan_replace(df,features_tabl_nan)
    pre.features_convert_types(df, features_convert, type_convert)
    X, y = turn.get_X_Y(df, features_used_X, feature_used_y)
    X = pre.features_dumnify(X, features_to_dumnify)
    if md == 'LogReg':
        model = modl.model_LogReg()
    elif md == 'RandFor':
        model = modl.model_RandFor()
    score = turn.model_learning(model, X, y)
    print("Score = ", score)
