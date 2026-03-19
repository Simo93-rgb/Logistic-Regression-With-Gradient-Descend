import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_score
from valutazione import evaluate_model, calculate_auc, calculate_auc
from logistic_regression_with_gradient_descend import LogisticRegressionGD
from ModelName import ModelName
from funzioni import fit_preprocess_train, transform_with_fitted_preprocess
import pandas as pd


def k_fold_cross_validation(
    X,
    y,
    model_enum,
    k=5,
    model_params=None,
    normalize=True,
    class_balancer="",
    corr=0.95
) -> tuple[dict, dict]:
    model_params = model_params or {}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    metrics_list = []
    sk_metrics_list = []

    is_pandas_input = hasattr(X, 'iloc')

    for train_index, val_index in kf.split(X):
        if is_pandas_input:
            X_train_raw, X_val_raw = X.iloc[train_index], X.iloc[val_index]
            y_train_raw, y_val_raw = y.iloc[train_index], y.iloc[val_index]
        else:
            X_train_raw, X_val_raw = X[train_index], X[val_index]
            y_train_raw, y_val_raw = y[train_index], y[val_index]

        X_train, y_train, preprocess_artifacts = fit_preprocess_train(
            X_train_raw,
            y_train_raw,
            normalize=normalize,
            class_balancer=class_balancer,
            corr=corr
        )
        X_val, y_val = transform_with_fitted_preprocess(X_val_raw, y_val_raw, preprocess_artifacts)

        model = LogisticRegressionGD(**model_params)
        sk_model = LogisticRegression(max_iter=model_params.get('n_iterations', 1000))

        model.fit(X_train, y_train)
        sk_model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        sk_predictions = sk_model.predict(X_val)

        # Valuta i modelli e accumula i risultati
        metrics = evaluate_model(model, X_val, predictions, y_val, model_enum.LOGISTIC_REGRESSION_GD.value)
        sk_metrics = evaluate_model(sk_model, X_val, sk_predictions, y_val, model_enum.SCIKIT_LEARN.value)

        metrics_list.append(metrics)
        sk_metrics_list.append(sk_metrics)

    # Converti metrics_list e sk_metrics_list in DataFrame Pandas
    metrics_df = pd.DataFrame(metrics_list)
    sk_metrics_df = pd.DataFrame(sk_metrics_list)

    # Identifica metriche numeriche e non numeriche dinamicamente
    numeric_metrics = metrics_df.select_dtypes(include=[np.number]).columns
    non_numeric_metrics = metrics_df.select_dtypes(exclude=[np.number]).columns

    # Calcola la media delle metriche numeriche con alta precisione
    mean_metrics = metrics_df[numeric_metrics].mean().apply(lambda x: round(x, 6)).to_dict()
    sk_mean_metrics = sk_metrics_df[numeric_metrics].mean().apply(lambda x: round(x, 6)).to_dict()

    # Ripristina i dizionari originali, mantenendo i campi non numerici invariati
    final_metrics_dict = {**mean_metrics, **metrics_df[non_numeric_metrics].iloc[0].to_dict()}
    final_sk_metrics_dict = {**sk_mean_metrics, **sk_metrics_df[non_numeric_metrics].iloc[0].to_dict()}

    return final_metrics_dict, final_sk_metrics_dict


def leave_one_out_cross_validation(X, y):
    model = LogisticRegressionGD()
    loo = LeaveOneOut()
    accuracies = []

    for train_index, val_index in loo.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        accuracy = accuracy_score(y_val, prediction)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy


# Funzione per Stratified K-Fold Cross-Validation
def stratified_k_fold_cross_validation(model, X_train, y_train, n_splits=5):
    stratified_kfold = StratifiedKFold(n_splits=n_splits)
    stratified_scores = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    return stratified_scores.mean()
