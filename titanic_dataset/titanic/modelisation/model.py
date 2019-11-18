from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def model_LogReg():
	print("------------------------------------------")
	print("Modele de Regression Logistique")
	return LogisticRegression()

def model_RandFor():
	print("------------------------------------------")
	print("Modele de Random Forest")
	return RandomForestClassifier()

def model_RandFor_param(max_d, min_simpl_l, min_simpl_splt, n_trees):
	return RandomForestClassifier(max_d, min_simpl_l, min_simpl_splt, n_trees)
