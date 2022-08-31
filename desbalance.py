
# _____________________________________________________________________________
# módulos                                                                  ####

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline as pipe

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report
import imblearn as im


# _____________________________________________________________________________
# apagadores                                                               ####

target_names = ['exterior', 'interior']

# _____________________________________________________________________________
# carga                                                                    ####

df = pd.read_csv('interior.csv')
df

# _____________________________________________________________________________
# preparación                                                              ####

# obtener lista de features categóricos
cat_var = ['zona', 'categoria']
df[cat_var] = (df[cat_var].
               apply(lambda x: pd.Series(x).
               astype('category')))

# lista features numéricos
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

# aplicar estratificación
X_train, X_test, y_train, y_test = tts(X,
                                       y,
                                       train_size=0.85,
                                       stratify=y,
                                       random_state=22)




# crear set de validación
X_train2, X_val, y_train2, y_val = tts(X_train,
                                       y_train,
                                       train_size=0.828969415701948,
                                       random_state=22)


print(round(len(X_train) / len(df), 2))
print(round(len(X_val) / len(df), 2))
print(round(len(X_test) / len(df), 2))


X_train2
y_train2


# _____________________________________________________________________________
# columns transformer                                                      ####

num_tr = Pipeline(steps=[
    ('scale', RobustScaler())])

cat_tr = Pipeline(steps=[
    ('one-hot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                              sparse=False))])

ct = ColumnTransformer(transformers=[
    ('num', num_tr, num_var),
    ('cat', cat_tr, cat_var)
])

ct.fit(X_train)
ct.transform(X_train)

# .............................................................................
# definir estimadores                                                      ####

logreg = LogisticRegression(random_state=22, max_iter=1000)

stratified_kfold = StratifiedKFold(n_splits=3,
                                   shuffle=True,
                                   random_state=22)

param_grid = {"rel__penalty": ["l1", "l2"]}

# .............................................................................
# instancias para balancear                                                ####

smt = SMOTE(random_state=22)

# .............................................................................
# evidenciar proceso de balanceo                                           ####

# balancear
x_trainb, y_trainb = smt.fit_resample(X_train.drop('categoria', axis=1),
                                      y_train)
df_balanceado = pd.DataFrame(y_trainb, columns=['interior'])

# gráfico de barra
sns.countplot(y='interior', data=df_balanceado, palette="Set2")
plt.show()

# .............................................................................
# pipeline imbalanced-learn                                                ####

con_smt = im.pipeline.Pipeline([
    ('preprocesador', ct),
    ('smt', smt),
    ('rel', logreg)
])

normal = pipe([('preprocesador', ct),  ('rel', logreg)])

sin_smt = im.pipeline.Pipeline([
    ('preprocesador', ct),
    ('rel', logreg)
])

param_grid = {'rel__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(estimator=con_smt,
                           param_grid=param_grid,
                           scoring='recall',
                           error_score='raise',
                           cv=stratified_kfold,
                           n_jobs=-1)

# _____________________________________________________________________________
# fit                                                                      ####

grid_search.fit(X_train, y_train)

# _____________________________________________________________________________
# métricas                                                                 ####

cv_score = grid_search.best_score_
test_score = grid_search.score(X_test, y_test)
print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')


y_pred = grid_search.predict(X_test)
print(classification_report(y_test,
                            y_pred,
                            zero_division=1,
                            target_names=target_names))

