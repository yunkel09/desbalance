
# _____________________________________________________________________________
# módulos                                                                  ####

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

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
                                       test_size=0.2,
                                       stratify=y,
                                       random_state=22)

X_train
y_train

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

logreg = LogisticRegression(random_state=22)

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

sin_smt = im.pipeline.Pipeline([
    ('preprocesador', ct),
    ('rel', logreg)
])


# _____________________________________________________________________________
# fit                                                                      ####

con_smt.fit(X_train, y_train)
y_hat_balanceado = con_smt.predict(X_test)

sin_smt.fit(X_train, y_train)
y_hat_imbalanceado = sin_smt.predict(X_test)

# _____________________________________________________________________________
# métricas                                                                 ####

print(classification_report(
    y_test,
    y_hat_balanceado, 
    zero_division=1,
    target_names=target_names))

print(classification_report(
    y_test,
    y_hat_imbalanceado,
    zero_division=1,
    target_names=target_names))
