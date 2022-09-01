
# _____________________________________________________________________________
# módulos                                                                  ####

import pandas as pd
import seaborn as sns
import imblearn as im
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold as rskf
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report

# _____________________________________________________________________________
# carga                                                                    ####

df = pd.read_csv('interior.csv')
df

# _____________________________________________________________________________
# preparación                                                              ####

# categóricos
cat_var = ['zona', 'bu']
df[cat_var] = (df[cat_var].
               apply(lambda x: pd.Series(x).
               astype('category')))

# numéricos
num_var = df.select_dtypes(exclude='category').columns.tolist()
num_var.remove('interior')
df.info()

# _____________________________________________________________________________
# exploración                                                              ####

# relación 1:100
df.interior.value_counts(normalize=True)

# gráfico de barra: desbalance
sns.countplot(y='interior', data=df, palette="Set2")
plt.show()

# _____________________________________________________________________________
# split                                                                    ####

X = df.drop('interior', axis=1)
y = df.interior

# train(70%), validación (15%), test (15%)

# aplicar estratificación
X_train, X_test, y_train, y_test = tts(X,
                                       y,
                                       train_size=0.85,
                                       stratify=y,
                                       random_state=22)

# crear set de validación a partir de set de entrenamiento
X_train, X_val, y_train, y_val = tts(X_train,
                                     y_train,
                                     train_size=0.828969415701948,
                                     random_state=22)

# data spending
tr = round(len(X_train) / len(df), 2)
va = round(len(X_val) / len(df), 2)
te = round(len(X_test) / len(df), 3)
print(f'trai: {tr}\nvali: {va}\ntest: {te}')

# _____________________________________________________________________________
# preprocesamiento                                                         ####

num_tr = Pipeline(steps=[
    ('scale', RobustScaler())])

cat_tr = Pipeline(steps=[
    ('one-hot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                              sparse=False))])

ct = ColumnTransformer(transformers=[
    ('num', num_tr, num_var),
    ('cat', cat_tr, cat_var)
])

# _____________________________________________________________________________
# validación cruzada                                                       ####

rep_stratified_kfold = rskf(n_splits=10,
                            n_repeats=5,
                            random_state=22)

# _____________________________________________________________________________
# estimadores                                                              ####

logreg = LogisticRegression(random_state=22, max_iter=1000)

# _____________________________________________________________________________
# balanceadores                                                            ####

smt = SMOTE(random_state=22)

# .............................................................................
# evidenciar proceso de balanceo                                           ####

# balancear
new_df = X_train.drop(cat_var, axis=1)
x_trainb, y_trainb = smt.fit_resample(new_df, y_train)
df_balanceado = pd.DataFrame(y_trainb, columns=['interior'])

# gráfico de barra balanceado
sns.countplot(y='interior', data=df_balanceado, palette="Set2")
plt.show()

# _____________________________________________________________________________
# pipelines                                                                ####

# con balanceo
con_smt = im.pipeline.Pipeline([
    ('preprocesador', ct),
    ('smt', smt),
    ('rel', logreg)
])

# sin balanceo
sin_smt = im.pipeline.Pipeline([
    ('preprocesador', ct),
    ('rel', logreg)
])

# _____________________________________________________________________________
# hiperparámetros                                                          ####

param_grid = {'rel__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# _____________________________________________________________________________
# cuadrícula                                                               ####

grid_search = GridSearchCV(estimator=con_smt,
                           param_grid=param_grid,
                           scoring='recall',
                           error_score='raise',
                           cv=rep_stratified_kfold,
                           n_jobs=-1)

# _____________________________________________________________________________
# fit                                                                      ####

grid_search.fit(X_train, y_train)

# _____________________________________________________________________________
# métricas                                                                 ####

# .............................................................................
# entrenamiento                                                            ####

cv_score = grid_search.best_score_
print(f'Cross-validation recall score: {cv_score}')

# .............................................................................
# validación                                                               ####

y_pred = grid_search.predict(X_val)
print(classification_report(y_val,
                            y_pred,
                            zero_division=1,
                            target_names=['exterior', 'interior']))

# .............................................................................
# prueba                                                                   ####

y_pred = grid_search.predict(X_test)
print(classification_report(y_test,
                            y_pred, 
                            zero_division=1))

# _____________________________________________________________________________
# End Of Script                                                            ####