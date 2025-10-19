import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
    matthews_corrcoef, precision_recall_curve, average_precision_score


# Funzione per valutare precision, recall e F1
def evaluate_model(model, X, predictions, y, model_name="", print_conf_matrix=False) -> dict:
    # Confusion Matrix
    conf_matrix = confusion_matrix(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    # False Positive Rate
    fp_rate = conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[0][0])  # FP / (FP + TN)
    # accuracy
    accuracy = (conf_matrix[1][1] + conf_matrix[0][0]) / len(predictions) # (TN + TF) / n
    # False Negative Rate
    fn_rate = conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[1][1])
    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y, predictions)
    cohen_kappa_score = cohen_kappa(y, predictions)

    # AUC (Area under ROC curve)
    auc = calculate_auc(model, X, y)

    # Precision-Recall AUC
    precision_recall_curve_auc = average_precision_score(y, model.predict_proba(X)[:, 1])

    if print_conf_matrix:
        # Trasforma in DataFrame per una visualizzazione più chiara
        cm_df = pd.DataFrame(conf_matrix, index=["B (Vera)", "M (Vera)"], columns=["B (Predetta)", "M (Predetta)"])
        # Stampa il DataFrame
        print(f'{model_name} Confusion Matrix')
        print(cm_df)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fp_rate': fp_rate,
        'fn_rate':fn_rate,
        'f1_score': f1,
        'mcc': mcc,
        'cohen_kappa': cohen_kappa_score,
        'auc': auc,
        'prc_auc': precision_recall_curve_auc,
        'model_name': model_name
    }


def cohen_kappa(y_val, y_pred):
    # Calcola la matrice di confusione
    conf_matrix = confusion_matrix(y_val, y_pred)

    # Totale dei campioni
    n = np.sum(conf_matrix)

    # Calcola le probabilità marginali (per le righe e le colonne della matrice di confusione)
    p0 = np.sum(np.diag(conf_matrix)) / n  # Accord osservato (percentuale di corrispondenza)

    # Frequenze marginali (righe e colonne)
    pe_rows = np.sum(conf_matrix, axis=1) / n
    pe_cols = np.sum(conf_matrix, axis=0) / n

    # Calcolo dell'accordo atteso
    pe = np.sum(pe_rows * pe_cols)

    # Formula di Cohen's Kappa
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 1.0

    return kappa


# Funzione per calcolare l'AUC (modello scikit-learn)
def calculate_auc(model, X_val, y_val):
    y_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_probs)
    return auc



