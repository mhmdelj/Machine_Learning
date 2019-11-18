import pandas as pd
import re

#fonction detecte les variables qui contiennent des valeurs manquantes
def features_nan_detect(df, nbr_entries):
    features_nan = []
    print("--------------------------------------------")
    print("les features qui contiennent des valeurs manquantes")
    for i in df.columns:
        x=df[i].isna().sum()
        if x != 0:
            features_nan.append(i)
            print(i, "---Nombre", x)
            print(i, "---Proportion", x/nbr_entries * 100)
    return features_nan

#fonction replace les valeurs manquantes par mediane/mode
def features_nan_replace(df, features_tabl):
    print("--------------------------------------------")
    print("Remplacement des valeurs manquantes")
    for i in features_tabl:
        for index, row in df.iterrows():
            if pd.isna(row[i]):
                t = df[i].dtype
                if t in ['float64', 'int64']:
                    df.loc[index, i] = df[i].median()
                else :
                    df.loc[index, i] = df[i].mode().iloc[0]
    print("Les valeurs manquantes sont remplacées par mode/mediane")

#fonction de conversion en string
def features_convert_types(df, to_convert, type_convert):
    print("--------------------------------------------")
    for i in to_convert:
        df[i] = df[i].apply(type_convert)
        print("Convert ", i," to ", type_convert)

#fonction de dumnification
def features_dumnify(x, to_dumnify):
    print("--------------------------------------------")
    print("Dumnification")
    x = pd.get_dummies(x, columns = to_dumnify)
    print(x)
    return x                    

def title_passenger(df):
    print("--------------------------------------------")
    print("Extration des titres des passagers")
    f = df['Name']
    titre = []
    for c in f:
        t = c.split(', ')[1].split('.')[0]
        titre.append(t)
    return titre