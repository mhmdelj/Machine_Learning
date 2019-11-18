import titanic.pipeline.pipeline as pip

file = "data/train.csv"
separator = ','
features_to_convert = ['Pclass', 'Parch']
type_convert = str
features_used_X = ['Pclass', 'SibSp', 'Parch']
feature_used_y = 'Survived'
features_to_dumnify = ['Parch', 'Pclass']
#Regression Logistique : LogReg
#RandomForest : RandFor
model = 'LogReg'


#pipeline_p(file, separator, features_to_convert, type_convert, features_used_X, feature_used_y, features_to_dumnify, model)
pip.pipeline_p(file, separator, features_to_convert, type_convert, features_used_X, feature_used_y, features_to_dumnify, model)