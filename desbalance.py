

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report
from sklearn import set_config
import imblearn as im

# _____________________________________________________________________________
# carga                                                                    ####

df = pd.read_csv('place.csv')

df

# _____________________________________________________________________________
# preparación                                                              ####

# convertir respuesta a variable indicadora (1: interior, 0: exterior)
df_clean = (pd.get_dummies(df,
                           columns=['lugar'],
                           drop_first=True).
            rename(columns={'lugar_INTERIOR': 'interior'}))

df_clean['interior'] = df_clean['interior']

# obtener lista de features por tipo de variable (num o cat)
df.info()
cat_var = ['zona', 'bu']
df_clean[cat_var] = (df_clean[cat_var].
                     apply(lambda x: pd.Series(x).
                     astype('category')))

num_var = df_clean.select_dtypes(exclude='category').columns.tolist()

num_var.remove('interior')

# _____________________________________________________________________________
# exploración                                                              ####

# relación 1:100
df_clean.interior.value_counts(normalize=True)

# gráfico de barra: desbalance
sns.countplot(y='interior', data=df_clean, palette="Set2")
plt.show()

# _____________________________________________________________________________
# split                                                                    ####

X = df_clean.drop('interior', axis=1)
y = df_clean.interior

# aplicar estratificación
X_train, X_test, y_train, y_test = tts(X,
                                       y,
                                       test_size=0.2,
                                       stratify=df[['lugar']],
                                       random_state=22)

# _____________________________________________________________________________
# recetas                                                                  ####

numeric_transformer = Pipeline(steps=[
    ('scale', RobustScaler())])

categorical_transformer = Pipeline(steps=[
    ('one-hot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                              sparse=False))])

ct = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_var),
    ('cat', categorical_transformer, cat_var)
])

ct.fit(X_train)
ct.transform(X_train)

# .............................................................................
# instancias SMOTE                                                         ####

smt = SMOTE(random_state=22)

# .............................................................................
# estimadores                                                              ####

logreg = LogisticRegression(random_state=22)
rfc = RandomForestClassifier(warm_start=True,
                             min_samples_leaf=5,
                             max_depth=10)
# .............................................................................
# pipeline                                                                 ####

clf = im.pipeline.Pipeline([
    ('preprocesador', ct),
    #('smt', smt),
    ('rfc', rfc)
])

# _____________________________________________________________________________
# fit                                                                      ####

clf.fit(X_train, y_train)

clf.score(X_train, y_train)
clf.score(X_test, y_test)

y_hat = clf.predict(X_test)
print(classification_report(y_test, y_hat))