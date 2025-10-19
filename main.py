import time

from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from funzioni import *
from validazione import *

if __name__ == "__main__":
    start_time = time.time()
    plotting = True
    k = 10
    param_file_path = 'Assets/best_parameters.json'
    # Carica e pre-processa i dati
    X, y = carica_dati()
    X_normalized, features_eliminate, y_encoded = preprocessa_dati(X, y, class_balancer="", corr=0.9)

    # Ottieni i nomi delle feature
    all_feature_names = X.columns
    remaining_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if
                               i not in features_eliminate]
    print(remaining_feature_names)

    # Suddivisione in train (80%), test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.3, random_state=42)

    # Caricamento iper parametri
    best_params, best_score = load_best_params()
    if not os.path.exists(param_file_path):
        # Eseguire l'ottimizzazione bayesiana se il file non esiste
        print("Eseguendo l'ottimizzazione bayesiana...")

        best_params, best_score = bayesian_optimization(
            X_normalized,
            y_encoded,
            scorer=make_scorer(false_negative_rate, greater_is_better=False)
            # scorer=make_scorer(precision_score, greater_is_better=True)
            # scorer=make_scorer(recall_score, greater_is_better=True)
            # scorer=make_scorer(matthews_corrcoef, greater_is_better=True)
        )
        save_best_params(best_params, param_file_path)
        print("Ottimizzazione bayesiana eseguita")
    print(f"Iperparametri caricati: {best_params}")

    # Cross-validation

    k_fold_metrics, k_fold_sk_metrics = k_fold_cross_validation(X_train, y_train, ModelName, k=k)
    print('Stampa delle metriche in fase di cross validazione')
    stampa_metriche_ordinate(k_fold_metrics, k_fold_sk_metrics, file_name="k_fold_metriche_definitivo")

    # Addestramento del modello
    model, sk_model = addestra_modelli(X_train, y_train, **best_params)

    model.plot_losses()
    # Valutazione finale
    print("\nValutazione finale sul Test Set:")
    test_predictions = model.predict(X_test)
    scores = evaluate_model(
        predictions=test_predictions,
        X=X_test,
        y=y_test,
        model=model,
        model_name=f"Modello {ModelName.LOGISTIC_REGRESSION_GD.value}",
        print_conf_matrix=True
    )

    test_sk_predictions = sk_model.predict(X_test)
    sk_scores = evaluate_model(
        predictions=test_sk_predictions,
        X=X_test,
        y=y_test,
        model=sk_model,
        model_name=f"Modello {ModelName.SCIKIT_LEARN.value}",
        print_conf_matrix=True
    )

    print('Stampa metriche dopo addestramento con X_test')
    stampa_metriche_ordinate(scores, sk_scores, save_to_file=True, file_name='metriche_modelli_test_definitivo')



    # Plotting
    if plotting:
        # plot_learning_curve_with_kfold(
        #     model=LogisticRegressionGD(**best_params),
        #     X=X_normalized,
        #     y=y_encoded,
        #     cv=k,
        #     model_name=ModelName.LOGISTIC_REGRESSION_GD.value
        # )
        # plot_learning_curve_with_kfold(
        #     model=LogisticRegression(max_iter=best_params.get('n_iterations', 1000)),
        #     X=X_normalized,
        #     y=y_encoded,
        #     cv=k,
        #     model_name=ModelName.SCIKIT_LEARN.value
        # )
        plot_graphs(X_train, y_train, y_test, test_predictions, test_sk_predictions, ModelName, remaining_feature_names)
        plot_results(X_test, y_test, model, sk_model, test_predictions, test_sk_predictions, scores,
                     sk_scores, ModelName)

    end_time = time.time()
    print(f"\nTempo di esecuzione totale: {end_time - start_time:.4f} secondi")
