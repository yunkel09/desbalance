
# _____________________________________________________________________________
# módulos                                                                  ####

# .............................................................................
# usuales                                                                  ####

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

# .............................................................................
# sklearn                                                                  ####

from sklearn.model_selection import (train_test_split as tts,
                                     RepeatedStratifiedKFold,
                                     RandomizedSearchCV)

# preprocesamiento
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

# modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# métricas
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from scipy.stats import loguniform

# .............................................................................
# tuberías                                                                 ####

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from imblearn.pipeline import Pipeline as imbPipeline

# .............................................................................
# muestreo                                                                 ####

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

# _____________________________________________________________________________
# opciones                                                                 ####

sns.set_palette("Set1")

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
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(y='interior', data=df).set(title='Desbalance de clases')
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
# estimadores                                                              ####

# logistic regression
lr = LogisticRegression(warm_start=True,
                        solver='lbfgs',
                        penalty='l2', 
                        max_iter=1000)

# random forest
rf = RandomForestClassifier()

# xgb
xgb = XGBClassifier(tree_method="hist", verbosity=0, silent=True)

# ensemble
lr_xgb_rf = VotingClassifier(estimators=[('lr', lr),
                                         ('xgb', xgb),
                                         ('rf', rf)],
                             voting='soft')

# _____________________________________________________________________________
# pipelines                                                                ####

# receta 1: sin balanceo
sin_muestreo = imbPipeline([
    # paso 1: aplicar transformaciones
    ('prep', ColumnTransformer([
        # paso 1.1: features numéricos
        ('num', make_pipeline(
             StandardScaler()),
         make_column_selector(dtype_include='number')
        ),
        # paso 1.2 features categóricos
        ('cat', make_pipeline(
            OneHotEncoder(sparse=False,
                          handle_unknown='infrequent_if_exist')),
         make_column_selector(dtype_include='category')
        )])
    ),
    # paso 3: consolidar estimador (ensemble)
    ('ensemble', lr_xgb_rf)
])

# receta 2: solo smote
standar_smote = imbPipeline([
    # paso 1: aplicar transformaciones
    ('prep', ColumnTransformer([
        # paso 1.1: features numéricos
        ('num', make_pipeline(
             RobustScaler()),
         make_column_selector(dtype_include='number')
        ),
        # paso 1.2 features categóricos
        ('cat',make_pipeline(
            OneHotEncoder(sparse=False, 
                          handle_unknown='infrequent_if_exist')),
         make_column_selector(dtype_include='category')
        )])
    ),
     # paso 3: muestreo
    ('smote', SMOTE(random_state=22)),
     # paso 4: consolidar estimador (ensemble)
    ('ensemble', lr_xgb_rf)
])

# receta 3: RandomUnderSampler
under_sampler = imbPipeline([
    # paso 1: aplicar transformaciones
    ('prep', ColumnTransformer([
        # paso 1.1: features numéricos
        ('num', make_pipeline(
            StandardScaler()),
         make_column_selector(dtype_include='number')
        ),
        # paso 1.2 features categóricos
        ('cat',make_pipeline(
            OneHotEncoder(sparse=False, 
                          handle_unknown='infrequent_if_exist')),
         make_column_selector(dtype_include='category')
        )])
    ),
    # paso 3: muestreo
    ('smote', RandomUnderSampler(random_state=22)),
    # paso 4: consolidar estimador (ensemble)
    ('ensemble', lr_xgb_rf)
])

# receta 4: sobre-muestreo con SMOTETomek
smote_tomek = imbPipeline([
    # paso 1: aplicar transformaciones
    ('prep', ColumnTransformer([
        # paso 1.1: features numéricos
        ('num', make_pipeline(
            StandardScaler()),
         make_column_selector(dtype_include='number')
        ),
        # paso 1.2 features categóricos
        ('cat',make_pipeline(
            OneHotEncoder(sparse=False, 
                          handle_unknown='infrequent_if_exist')),
         make_column_selector(dtype_include='category')
        )])
    ),
    # paso 3: muestreo
    ('smote', SMOTETomek(random_state=22)),
    # paso 4: consolidar estimador (ensemble)
    ('ensemble', lr_xgb_rf)
])


tuberias = [sin_muestreo,
            standar_smote,
            under_sampler,
            smote_tomek]

pipe_dict = {0: 'sin_balanceo',
             1: 'smote_simple',
             2: 'sub_muestreo',
             3: 'smotetomek'}

# _____________________________________________________________________________
# cuadrícula de hiperparámetros                                            ####

params = {
    # 'ensemble__lr__solver': ['newton-cg', 'lbfgs', 'liblinear'],
    # 'ensemble__lr__penalty': ['l2'],
    'ensemble__lr__C': loguniform(1e-5, 100),
    'ensemble__xgb__learning_rate': [0.1],
    'ensemble__xgb__max_depth': [7, 10, 15, 20],
    'ensemble__xgb__min_child_weight': [10, 15, 20, 25],
    'ensemble__xgb__colsample_bytree': [0.8, 0.9, 1],
    'ensemble__xgb__n_estimators': [300, 400, 500, 600],
    'ensemble__xgb__reg_alpha': [0.5, 0.2, 1],
    'ensemble__xgb__reg_lambda': [2, 3, 5],
    'ensemble__xgb__gamma': [1, 2, 3],
    'ensemble__rf__max_depth': [7, 10, 15, 20],
    'ensemble__rf__min_samples_leaf': [1, 2, 4],
    'ensemble__rf__min_samples_split': [2, 5, 10],
    'ensemble__rf__n_estimators': [300, 400, 500, 600],
}

# _____________________________________________________________________________
# validación cruzada                                                       ####

rsf = RepeatedStratifiedKFold(n_splits=10,
                              n_repeats=5,
                              random_state=22)

cdx = []
for pipe in tuberias:
    cdx.append(RandomizedSearchCV(estimator=pipe,
                                  param_distributions=params,
                                  scoring='recall',
                                  error_score='raise',
                                  verbose=2,
                                  n_jobs=-1,
                                  cv=rsf))

# _____________________________________________________________________________
# fit                                                                      ####

# training time:
# 10 folds & 10 reps: 145 seg (2.41 min) con 12 nucleos físicos en paralelo
start = time.time()
for pipe in cdx:
    pipe.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

# _____________________________________________________________________________
# métricas                                                                 ####

dxl = []
for idx, val in enumerate(cdx):

    y_pred = val.predict(X_val)

    val_recall = recall_score(y_true=y_val,
                              y_pred=y_pred,
                              pos_label=1,
                              zero_division=1,
                              average='binary')

    val_precision = precision_score(y_true=y_val,
                                    y_pred=y_pred,
                                    pos_label=1,
                                    zero_division=1,
                                    average='binary')

    val_f1 = f1_score(y_true=y_val,
                      y_pred=y_pred,
                      pos_label=1,
                      zero_division=1,
                      average='binary')

    dxl.append([pipe_dict[idx],
                val.best_score_,
                val_recall,
                val_precision,
                val_f1])


# .............................................................................
# métricas                                                   ####

dfx = pd.DataFrame(dxl, columns=['modelo',
                                 'cv_score',     # recall
                                 'recall',
                                 'precision',
                                 'f1_score'])
dfx['i'] = dfx.index

# cv_score y validación
dfx

# .............................................................................
# seleccionar mejor modelo                                                 ####

dfp = pd.melt(dfx, id_vars=['modelo'], value_vars=['recall', 'precision'])

# gráfica de barras
(sns.barplot(x='value',
             y='modelo',
             hue='variable',
             data=dfp,
             palette=color).
 set(title='Precision y Recall por Modelo'))
plt.show()

# seleccionar el mejor modelo con base a f1-score
mejor_modelo = (dfx[dfx.f1_score == dfx.f1_score.max()].
                head(1).i.item())

# revisar cual es el mejor modelo
pipe_dict[mejor_modelo]

# .............................................................................
# prueba                                                                   ####

prediccion = cdx[mejor_modelo].predict(X_test)
print(classification_report(y_test,
                            prediccion,
                            zero_division=1,
                            target_names=['exterior', 'interior']))

# .............................................................................
# mejores hiperparámetros                                                  ####

best_params = cdx[1].best_params_
hiper = pd.DataFrame.from_dict(best_params, orient='index')
hiper

# _____________________________________________________________________________
# gráfica                                                                  ####





# _____________________________________________________________________________
# End Of Script                                                            ####


