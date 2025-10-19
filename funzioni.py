import time

import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from validazione import k_fold_cross_validation, leave_one_out_cross_validation, stratified_k_fold_cross_validation
from logistic_regression_with_gradient_descend import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from skopt import BayesSearchCV
from plot import *
import os
import json


def carica_dati(file_path='Assets/dataset', file_name='breast_cancer_wisconsin'):
    if not os.path.exists(f'{file_path}/{file_name}.csv'):
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features
        y = dataset.data.targets
        # Se esiste una colonna chiamata 'ID', la eliminiamo
        if 'ID' in X.columns:
            X = X.drop(columns=['ID'])
        # Crea un DataFrame unendo X_normalized e y_encoded
        df = pd.DataFrame(X, columns=X.columns)
        df['target'] = y
        # Salva in CSV
        csv_file = os.path.join(file_path, f'{file_name}.csv')
        df.to_csv(csv_file, index=False)
        print(f"Dataset salvato in {csv_file}")
    else:
        df = pd.read_csv(f'{file_path}/{file_name}.csv')
        X = df.drop(columns='target')
        y = df['target']
    return X, y


def preprocessa_dati(X, y, normalize=True, class_balancer="", corr=0.95, save_dataset=False, file_path='Assets/dataset'):
    # Elimina le feature altamente correlate
    X_df = X

    # Imputazione dei NaN e normalizzazione
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Z-Score Normalization
    X = (X - X.mean(axis=0)) / X.std(axis=0) if normalize else X

    X, features_eliminate = elimina_feature_correlate(X, soglia=corr)
    print(f"Features eliminate: {features_eliminate}")
    # Ottieni i nomi delle feature
    all_feature_names = X_df.columns
    remaining_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if
                               i not in features_eliminate]
    # print(remaining_feature_names)

    
    # Encoding delle classi
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).ravel()

    # Plotting delle classi prima del bilanciamento
    # plot_class_distribution(y_encoded, file_name='class_distribution_pie_breast_cancer')

    # Gestione del bilanciamento delle classi
    if class_balancer:
        resampler = SMOTE(random_state=42) if class_balancer == "SMOTE" else RandomUnderSampler(random_state=42)
        X, y_encoded = resampler.fit_resample(X, y_encoded)
        plot_class_distribution(y_encoded, file_name=f'class_distribution_pie_breast_cancer_{class_balancer}')
    # Mescola i dati dopo il resampling
    X, y_encoded = shuffle(X, y_encoded, random_state=42)

    if save_dataset:
        file_name = 'breast_cancer_wisconsin_edited'
        # Crea un DataFrame unendo X_normalized e y_encoded
        df = pd.DataFrame(X, columns=remaining_feature_names)
        df['target'] = y_encoded
        # Salva in CSV
        csv_file = os.path.join(file_path, f'{file_name}.csv')
        df.to_csv(csv_file, index=False)
        print(f"Dataset salvato in {csv_file}")

    return X, features_eliminate, y_encoded


def elimina_feature_correlate(X:np.ndarray, soglia=0.95):
    """
    Elimina feature altamente correlate da un array numpy X basandosi sulla soglia fornita.

    Parametri:
    - X: array numpy bidimensionale, dove le colonne sono le feature.
    - soglia: valore soglia per la correlazione (default 0.95).

    Restituisce:
    - X_ridotto: array numpy con le feature eliminate.
    - feature_da_eliminare: set degli indici delle feature eliminate.
    """
    # Calcola la matrice di correlazione
    corr_matrix = np.corrcoef(X, rowvar=False)
    plot_corr_matrix(corr_matrix)
    # Calcola la varianza di ogni feature
    feature_variances = np.var(X, axis=0)

    num_features = corr_matrix.shape[0]
    feature_da_eliminare = set()

    for i in range(num_features):
        if i in feature_da_eliminare:
            continue  # Salta le feature già eliminate
        for j in range(i + 1, num_features):
            if j in feature_da_eliminare:
                continue  # Salta le feature già eliminate
            if abs(corr_matrix[i, j]) >= soglia:
                # Confronta le varianze per decidere quale eliminare
                if feature_variances[i] < feature_variances[j]:
                    feature_da_eliminare.add(i)
                    break  # Esci dal loop interno se la feature i è eliminata
                else:
                    feature_da_eliminare.add(j)

    # Elimina le feature dal dataset
    X_ridotto = np.delete(X, list(feature_da_eliminare), axis=1)

    return X_ridotto, feature_da_eliminare


def addestra_modelli(X_train, y_train, **best_params):
    start = time.time()
    # Modello Logistic Regression implementato
    model = LogisticRegressionGD()
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    end = time.time()
    print(f'#################################\nTempo addestramento mio modello: {end-start}\n#################################')
    # Modello di scikit-learn Logistic Regression
    start = time.time()
    sk_model = LogisticRegression(max_iter=best_params.get('n_iterations', 1000))
    sk_model.fit(X_train, y_train)
    end = time.time()
    print(f'#################################\nTempo addestramento modello sklearn: {end-start}\n#################################')

    return model, sk_model


def bayesian_optimization(X_train, y_train, scorer=None):
    if not scorer:
        scorer = make_scorer(false_negative_rate, greater_is_better=False)
    # Definisci lo spazio degli iperparametri da ottimizzare
    param_space = {
        'learning_rate': (0.001, 0.1, 'log-uniform'),
        'lambda_': (1e-4, 0.1, 'log-uniform'),
        'n_iterations': (1000, 10000),
        'regularization': ['none', 'ridge', 'lasso']
    }

    bayes_search = BayesSearchCV(
        estimator=LogisticRegressionGD(),
        search_spaces=param_space,
        n_iter=25,
        cv=10,
        scoring=scorer,
        n_jobs=-1,
        random_state=42
    )

    bayes_search.fit(X_train, y_train)

    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    return best_params, best_score


def save_best_params(best_params, file_path="best_parameters.json"):
    with open(file_path, 'w') as file:
        json.dump(best_params, file)
    print(f"Parametri salvati in {file_path}.")


def load_best_params(X_train=None, y_train=None, file_path="Assets/best_parameters.json"):
    # Controllo se esiste il file con i parametri salvati
    if os.path.exists(file_path):
        print(f"Caricamento dei parametri ottimali da {file_path}...")
        with open(file_path, 'r') as file:
            best_params = json.load(file)
        best_score = best_params.pop("accuracy", None)  # Rimuovi 'accuracy' se presente
    else:
        best_params = {
            "lambda_": 0.0,
            "learning_rate": 0.1,
            "n_iterations": 1000,
            "regularization": "none"
        }
        best_score = None

    return best_params, best_score


def stampa_metriche_ordinate(metriche_modello1, metriche_modello2, file_path="Assets/", save_to_file=True,
                             file_name=""):
    # Creazione della lista delle metriche
    lista_metriche = [metriche_modello1, metriche_modello2]
    for metriche in lista_metriche:
        for chiave, valore in metriche.items():
            if isinstance(valore, (int, float)):  # Verifica se il valore è numerico
                metriche[chiave] = round(valore, 6)  # Arrotonda a 6 cifre decimali

    # Creazione del DataFrame escludendo 'conf_matrix'
    df_metriche = pd.DataFrame(lista_metriche).set_index('model_name')  # .drop(columns=['conf_matrix'])

    # Ordinare le colonne se necessario
    df_metriche = df_metriche[sorted(df_metriche.columns)]

    # Stampare il DataFrame
    print(df_metriche)

    # Salvataggio su file
    # Controlla se la directory esiste, altrimenti la crea
    if save_to_file:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Salva il DataFrame in un file CSV
        csv_file = os.path.join(file_path, f'{file_name}.csv' if file_name else "metriche_modelli.csv")
        json_file = os.path.join(file_path, f'{file_name}.json' if file_name else "metriche_modelli.json")
        df_metriche.to_csv(csv_file)
        df_metriche.to_json(json_file)


def false_negative_penalty(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return -fn  # Penalizza fortemente i falsi negativi (obiettivo minimizzazione)


def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)  # Calcolo della FNR
    return fnr
