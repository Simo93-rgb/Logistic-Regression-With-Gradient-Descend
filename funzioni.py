import time

import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from logistic_regression_with_gradient_descend import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from plot import *
import os
import json
from ModelName import ModelName


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
    raise RuntimeError(
        "preprocessa_dati e disabilitata per prevenire data leakage. "
        "Usa fit_preprocess_train(...) e transform_with_fitted_preprocess(...)."
    )


def fit_preprocess_train(X_train, y_train, normalize=True, class_balancer="", corr=0.95):
    """
    Fitta il preprocessing solo sul training set e restituisce gli artefatti
    necessari a trasformare il test set senza leakage.
    """
    X_df = X_train.copy()

    imputer = SimpleImputer(strategy='mean')
    X_train_processed = imputer.fit_transform(X_train)

    train_mean = X_train_processed.mean(axis=0)
    train_std = X_train_processed.std(axis=0)
    train_std[train_std == 0] = 1.0

    if normalize:
        X_train_processed = (X_train_processed - train_mean) / train_std

    X_train_processed, features_eliminate = elimina_feature_correlate(X_train_processed, soglia=corr)
    # print(f"Features eliminate: {features_eliminate}")

    all_feature_names = X_df.columns
    remaining_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if
                               i not in features_eliminate]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train).ravel()

    if class_balancer:
        resampler = SMOTE(random_state=42) if class_balancer == "SMOTE" else RandomUnderSampler(random_state=42)
        X_train_processed, y_train_encoded = resampler.fit_resample(X_train_processed, y_train_encoded)
        plot_class_distribution(y_train_encoded, file_name=f'class_distribution_pie_breast_cancer_{class_balancer}')

    X_train_processed, y_train_encoded = shuffle(X_train_processed, y_train_encoded, random_state=42)

    preprocess_artifacts = {
        'imputer': imputer,
        'normalize': normalize,
        'train_mean': train_mean,
        'train_std': train_std,
        'features_eliminate': features_eliminate,
        'label_encoder': label_encoder,
        'remaining_feature_names': remaining_feature_names
    }

    return X_train_processed, y_train_encoded, preprocess_artifacts


def transform_with_fitted_preprocess(X, y, preprocess_artifacts):
    """
    Applica al dataset in input le trasformazioni fittate sul training set.
    """
    X_processed = preprocess_artifacts['imputer'].transform(X)

    if preprocess_artifacts['normalize']:
        X_processed = (X_processed - preprocess_artifacts['train_mean']) / preprocess_artifacts['train_std']

    features_eliminate = list(preprocess_artifacts['features_eliminate'])
    X_processed = np.delete(X_processed, features_eliminate, axis=1)

    y_encoded = preprocess_artifacts['label_encoder'].transform(y).ravel()

    return X_processed, y_encoded


def elimina_feature_correlate(X:np.ndarray, soglia=0.95, plot_matrix=False):
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
    if plot_matrix:
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


def bayesian_optimization(
    X_train_raw,
    y_train_raw,
    scorer=None,
    cv=10,
    class_balancer="",
    corr=0.95,
    n_iter=25
):
    """
    Ottimizzazione bayesiana leak-safe: ogni valutazione usa CV con preprocessing
    rifittato nel train di ciascun fold.
    """
    from validazione import k_fold_cross_validation

    metric_name_by_scorer = {
        'false_negative_rate': 'fn_rate',
        'matthews_corrcoef': 'mcc',
        'precision_score': 'precision',
        'recall_score': 'recall',
        'f1_score': 'f1_score',
        'accuracy_score': 'accuracy',
        'roc_auc_score': 'auc'
    }

    # Obiettivo di default: minimizzare i falsi negativi.
    selected_metric = 'recall'
    greater_is_better = True

    if scorer is not None and hasattr(scorer, '_score_func'):
        scorer_name = scorer._score_func.__name__
        selected_metric = metric_name_by_scorer.get(scorer_name, selected_metric)
        greater_is_better = getattr(scorer, '_sign', -1) > 0

    dimensions = [
        Real(0.001, 0.1, prior='log-uniform', name='learning_rate'),
        Real(1e-4, 0.1, prior='log-uniform', name='lambda_'),
        Integer(1000, 10000, name='n_iterations'),
        Categorical(['none', 'ridge', 'lasso'], name='regularization')
    ]

    @use_named_args(dimensions)
    def objective(**params):
        custom_metrics, _ = k_fold_cross_validation(
            X_train_raw,
            y_train_raw,
            ModelName,
            k=cv,
            model_params=params,
            class_balancer=class_balancer,
            corr=corr
        )
        metric_value = custom_metrics[selected_metric]
        # gp_minimize minimizza: invertiamo il segno per metriche da massimizzare.
        return -metric_value if greater_is_better else metric_value

    result = gp_minimize(
        objective,
        dimensions,
        n_calls=n_iter,
        random_state=42
    )

    best_params = {
        dimensions[i].name: (result.x[i].item() if isinstance(result.x[i], np.generic) else result.x[i])
        for i in range(len(dimensions))
    }
    best_score = result.fun
    return best_params, best_score


def save_best_params(best_params, file_path="best_parameters.json"):
    def _json_converter(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(file_path, 'w') as file:
        json.dump(best_params, file, indent=2, default=_json_converter)
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
