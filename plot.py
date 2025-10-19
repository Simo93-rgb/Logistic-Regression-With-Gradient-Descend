import os

import numpy
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc, precision_score, \
    f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

from logistic_regression_with_gradient_descend import LogisticRegressionGD


def plot_sigmoid():
    x_dim, y_dim = [16, 12]
    # Crea un array di valori x
    x = np.linspace(-10, 10, 100)

    # Funzione sigmoidale
    sigmoid = 1 / (1 + np.exp(-x))

    # Plot
    plt.figure(figsize=(x_dim, y_dim))
    plt.plot(x, sigmoid, color='blue')
    plt.title("Funzione Sigmoidale", fontsize=24)
    plt.xlabel('z', fontsize=18)
    plt.ylabel('Sigmoid(z)', fontsize=18)
    plt.grid(True)
    plt.savefig('Assets/sigmoid_function.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_corr_matrix(corr_matrix: numpy.ndarray, features_eliminate=None):
    x_dim, y_dim = [20, 16]

    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12})  # Font migliorato
    plt.title("Matrice di Correlazione delle Feature", fontsize=24)

    # Evidenzia le feature eliminate
    if features_eliminate:
        for feature in features_eliminate:
            plt.gca().add_patch(plt.Rectangle((feature, feature), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.savefig('Assets/correlation_matrix_breast_cancer.png', format='png', dpi=600,
                bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_class_distribution(y, file_name=""):
    x_dim, y_dim = [16, 12]

    # Conta il numero di campioni per ciascuna classe
    unique, counts = np.unique(y, return_counts=True)

    # Calcola le percentuali
    total = np.sum(counts)
    percentages = [(count / total) * 100 for count in counts]

    # Etichette con la percentuale e il numero di elementi
    labels = [f'B (Benigno) - {counts[0]} ({percentages[0]:.1f}%)',
              f'M (Maligno) - {counts[1]} ({percentages[1]:.1f}%)']

    # Crea un grafico a torta
    plt.figure(figsize=(x_dim, y_dim))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'orange'], startangle=90,
            textprops={'fontsize': 27})  # Font ingrandito del 50% (18 * 1.5 = 27)

    plt.title('Distribuzione delle Classi', fontsize=36)  # Font titolo ingrandito del 50% (24 * 1.5 = 36)
    plt.axis('equal')  # Assicura che il grafico sia disegnato come un cerchio

    # Salva l'immagine, se specificato
    if file_name:
        plt.savefig(f'Assets/{file_name}.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata

    # Mostra il grafico
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    x_dim, y_dim = [16, 12]
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Etichette per le classi
    class_names = ['B', 'M']

    plt.figure(figsize=(x_dim, y_dim))

    # Heatmap con font migliorato
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 20})  # Ingrandisci i numeri nelle celle

    # Aumentare la dimensione dei font per titolo, etichette degli assi e tick
    plt.title(f"Matrice di Confusione - {model_name}", fontsize=28, weight='bold')  # Titolo più grande e in grassetto
    plt.xlabel("Predicted", fontsize=24, weight='bold')  # Etichetta asse X più grande e in grassetto
    plt.ylabel("True", fontsize=24, weight='bold')  # Etichetta asse Y più grande e in grassetto

    # Aumenta le dimensioni delle etichette degli assi (tick labels)
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')

    # Salva il plot con maggiore risoluzione
    plt.savefig(f'Assets/confusion_matrix_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()



def plot_roc_curve(y_true, y_probs, model_name):
    x_dim, y_dim = [16, 12]
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(x_dim, y_dim))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f"Curva ROC - {model_name}", fontsize=24)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(f'Assets/ROC_curve_{model_name}.png', format='png', dpi=600,
                bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_roc_curve_sklearn(model, X_val, y_val, model_name=""):
    x_dim, y_dim = [16, 12]
    y_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    auc = roc_auc_score(y_val, y_probs)

    plt.figure(figsize=(x_dim, y_dim))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve {model_name} (AUC = {auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'Curva ROC - {model_name}', fontsize=24)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(f'Assets/ROC_curve_{model_name}.png', format='png', dpi=600,
                bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_metrics_comparison(metrics_dict: dict, model_names):
    x_dim, y_dim = 16, 9

    # Ottieni le chiavi (metriche) dal primo modello
    metrics = [metric for metric in metrics_dict[model_names[0]].keys() if metric != 'model_name']  # Escludiamo 'model_name'

    fig, ax = plt.subplots(figsize=(x_dim, y_dim))

    bar_width = 0.35
    index = np.arange(len(metrics))

    # Trova il minimo e il massimo delle metriche per adattare i limiti dell'asse y
    all_results = [metrics_dict[model_name][metric] for model_name in model_names for metric in metrics]
    min_metric = min(all_results)
    max_metric = max(all_results)

    # Definisci i limiti dell'asse y per fare zoom solo sulla parte alta delle barre
    zoom_factor = 0.1  # Puoi regolare questo fattore per aumentare/diminuire lo zoom
    y_min = min_metric - zoom_factor * (max_metric - min_metric)
    y_max = max_metric + zoom_factor * (max_metric - min_metric)

    for i, model_name in enumerate(model_names):
        # Ottieni i valori delle metriche per ciascun modello
        results = [metrics_dict[model_name][metric] for metric in metrics]
        ax.bar(index + i * bar_width, results, bar_width, label=model_name)

    ax.set_xlabel('Metriche', fontsize=18)
    ax.set_title('Confronto delle Metriche tra i Modelli', fontsize=24)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics, fontsize=16, rotation=45)  # Ruota leggermente le etichette se sono tante
    ax.legend(fontsize=16)

    # Imposta i limiti dell'asse y per il "zoom"
    ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    plt.savefig('Assets/metrics_comparison.png', format='png', dpi=600, bbox_inches='tight')  # Risoluzione migliorata
    plt.show()



def plot_regularization_effect(X_train, y_train, feature_names, lambdas, regularization_type='ridge'):
    x_dim, y_dim = [20, 12]
    coefficients = []

    for _lambda in lambdas:
        model = LogisticRegressionGD(learning_rate=0.01, n_iterations=1000, lambda_=_lambda,
                                     regularization=regularization_type)
        model.fit(X_train, y_train)
        coefficients.append(model.theta)

    coefficients = np.array(coefficients)

    plt.figure(figsize=(x_dim, y_dim))
    for i in range(coefficients.shape[1]):
        plt.plot(lambdas, coefficients[:, i], label=f'{feature_names[i]}', lw=2)

    plt.xscale('log')
    plt.title(f"Effetto della Regolarizzazione {regularization_type.capitalize()} sui Coefficienti", fontsize=24)
    plt.xlabel("Lambda", fontsize=18)
    plt.ylabel("Coefficients", fontsize=18)
    plt.legend(loc='upper left', fontsize=16)
    plt.savefig(f'Assets/regularization_effect_{regularization_type}.png', format='png', dpi=600,
                bbox_inches='tight')  # Risoluzione migliorata
    plt.show()


def plot_precision_recall(y_true, y_scores, model_name="", save_file=True):
    # Calcola precisione, recall e soglie
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Calcola l'AUC (Area Under the Curve) per la curva Precision-Recall
    auc_score = auc(recalls, precisions)

    # Crea il plot
    plt.figure(figsize=(16, 12))
    plt.plot(recalls, precisions, label=f'AUC = {auc_score:.3f}', linewidth=2)

    # Impostazioni del grafico
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title(f'Curva Precision-Recall: {model_name}', fontsize=24)
    plt.legend(loc='lower left', fontsize=16)
    plt.grid(True)

    # Salva il grafico, se richiesto
    if save_file:
        plt.savefig(f'Assets/precision_vs_recall_{model_name}.png', format='png', dpi=600, bbox_inches='tight')

    # Mostra il grafico
    plt.show()


def plot_learning_curve_with_kfold(model, X, y, cv=5, model_name="", scoring='neg_log_loss', fig_size=(16, 12)):
    """
    Plotta la learning curve utilizzando k-fold cross-validation.

    Parameters:
    - model: Il modello scikit-learn da valutare.
    - X: Matrice delle feature.
    - y: Vettore target.
    - cv: Numero di fold per la cross-validation (default=5).
    - model_name: Nome del modello per il titolo del grafico e il file (default="").
    - scoring: Metrica di scoring (default='neg_log_loss').
    - fig_size: Dimensioni del grafico (default=(16, 12)).

    Returns:
    - None. Mostra e salva il grafico della learning curve.
    """
    # Se model_name non è fornito, ottieni il nome del modello
    if not model_name:
        model_name = type(model).__name__

    # Frazioni del training set da usare (es. dal 10% al 100%)
    train_sizes = np.linspace(0.1, 1.0, cv)

    # Calcola learning curve con K-Fold CV
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=-1
    )

    # Calcola la media e deviazione standard delle performance
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Invertiamo il segno se usiamo una metrica negativa (come 'neg_log_loss')
    if scoring in ['neg_log_loss', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
        train_mean *= -1
        train_std *= -1
        val_mean *= -1
        val_std *= -1

    # Plot della learning curve
    plt.figure(figsize=fig_size)
    plt.plot(train_sizes, train_mean, label="Training Error", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.plot(train_sizes, val_mean, label="Cross-validation Error", color="orange")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    plt.ylim(0, 0.5)
    plt.title(f"Learning Curve: {model_name}", fontsize=24)
    plt.xlabel("Number of Training Samples", fontsize=18)
    plt.ylabel("Error (Cost Function)", fontsize=18)
    plt.legend(loc="best", fontsize=16)
    plt.grid(True)

    # Assicurati che la directory 'Assets' esista
    if model_name:
        os.makedirs('Assets', exist_ok=True)
        plt.savefig(f'Assets/learning_curve_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_learning_curve_with_loss(estimator, X_train, y_train, cv=10, train_sizes=np.linspace(0.1, 1.0, 10),
                                  scoring='neg_log_loss', model_name=""):
    """
    Funzione per plottare la curva di apprendimento con la loss sui dati di training e validation.

    Parametri:
    - estimator: il modello (es. LogisticRegression)
    - X_train: dataset di training (feature)
    - y_train: dataset di training (target)
    - cv: numero di fold per la cross-validation
    - train_sizes: frazioni di training set da usare per calcolare il punteggio
    - scoring: metrica di valutazione ('neg_log_loss' per la loss)

    Output:
    - Grafico delle curve di apprendimento per il training e il validation set con la loss.
    """
    plt.figure(figsize=(10, 6))

    # Calcola le curve di apprendimento
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X_train, y_train, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=-1
    )

    # Convertiamo i punteggi negativi delle loss in valori positivi (perchè scikit-learn usa score, non loss)
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    # Plottiamo i risultati
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training loss")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Validation loss")

    # Plottiamo anche le aree di deviazione standard
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="r",
                     alpha=0.2)
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, color="g", alpha=0.2)

    plt.title(f"Curva di Apprendimento (Loss) {model_name}")
    plt.xlabel("Numero di campioni di training")
    plt.ylabel("Loss (Log-Loss)")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def plot_results(X_test, y_test, model, sk_model, test_predictions, test_sk_predictions, scores, sk_scores, model_enum):
    auc = scores['auc']
    sk_auc = sk_scores['auc']
    # Curve ROC
    y_probs = model.sigmoid(np.dot(X_test, model.theta) + model.bias)
    plot_roc_curve(y_test, y_probs, f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}")

    y_sk_probs = sk_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_sk_probs, f"Modello {model_enum.SCIKIT_LEARN.value}")

    plot_prc_auc(
        model,
        X_test,
        y_test,
        model_name=f"modello_{model_enum.LOGISTIC_REGRESSION_GD.value}"
    )
    plot_prc_auc(
        sk_model,
        X_test,
        y_test,
        model_name=f"modello_{model_enum.SCIKIT_LEARN.value}"
    )
    # plot_precision_recall(y_test, test_predictions, model_name=f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}")
    # plot_precision_recall(y_test, test_sk_predictions, model_name=f"Modello {model_enum.SCIKIT_LEARN.value}")
    logistic = scores.pop('model_name')
    sk_logistic = sk_scores.pop('model_name')
    # Confronto delle metriche
    metrics_dict = {
        logistic: scores,
        sk_logistic: sk_scores
    }
    scores['model_name'] = logistic
    sk_scores['model_name'] = sk_logistic
    plot_metrics_comparison(metrics_dict, [f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}",
                                           f"Modello {model_enum.SCIKIT_LEARN.value}"])




def plot_prc_auc(model, X_test, y_test, model_name="", fig_size=(10, 8), save_file=True, num_thresholds=10):
    """
    Plotta la curva Precision-Recall e calcola l'AUC per un modello scikit-learn.

    Parameters:
    - model: Il modello scikit-learn da valutare.
    - X_test: Matrice delle feature di test.
    - y_test: Vettore target di test.
    - model_name: Nome del modello per il titolo del grafico (default="").
    - fig_size: Dimensioni del grafico (default=(10, 8)).
    - save_file: Se True, salva il grafico come file PNG.
    - num_thresholds: Numero di soglie da rappresentare sul grafico.

    Returns:
    - None. Mostra il grafico della curva Precision-Recall.
    """
    # Prevedi le probabilità per la classe positiva
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcola precisione, recall e soglie
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # Calcola l'AUC della curva PRC
    prc_auc = auc(recall, precision)

    # Plot della curva Precision-Recall
    plt.figure(figsize=fig_size)
    plt.plot(recall, precision, label=f'PRC AUC = {prc_auc:.2f}', color='blue')

    # Configura il grafico
    plt.title(f'Precision-Recall Curve: {model_name}', fontsize=18)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)

    # Seleziona soglie rappresentative per evidenziare alcuni punti
    # Selezioniamo num_thresholds punti uniformemente distribuiti
    thresholds_to_plot = np.linspace(0, len(thresholds) - 1, num=num_thresholds, dtype=int)

    # Plotta punti rappresentativi con soglie
    for idx in thresholds_to_plot:
        plt.scatter(recall[idx], precision[idx], marker='o', color='red')
        plt.text(recall[idx], precision[idx], f'{thresholds[idx]:.2f}', fontsize=10, ha='right')

    if save_file:
        plt.savefig(f'Assets/prc_auc_{model_name}.png', format='png', dpi=600, bbox_inches='tight')

    # Mostra il grafico
    plt.show()



def plot_graphs(X_train, y_train, y_test, test_predictions, test_sk_predictions, model_enum, remaining_feature_names):
    # Plottare la funzione sigmoidale
    plot_sigmoid()

    # Matrici di confusione
    plot_confusion_matrix(y_test, test_predictions, f"Modello {model_enum.LOGISTIC_REGRESSION_GD.value}")
    plot_confusion_matrix(y_test, test_sk_predictions, f"Modello {model_enum.SCIKIT_LEARN.value}")

    # Plottare l'effetto della regolarizzazione
    lambdas = np.logspace(-4, 2, 100)  # Lista di valori di lambda su scala logaritmica
    plot_regularization_effect(X_train, feature_names=remaining_feature_names, y_train=y_train, lambdas=lambdas,
                               regularization_type='ridge')
    plot_regularization_effect(X_train, feature_names=remaining_feature_names, y_train=y_train, lambdas=lambdas,
                               regularization_type='lasso')
