import time

from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from src.funzioni import *
from src.validazione import *

if __name__ == "__main__":
    start_time = time.time()
    plotting = True
    k = 10
    param_file_path = 'assets/best_parameters.json'
    class_balancer = ""
    corr_threshold = 0.9

    # Carica dati grezzi
    X, y = carica_dati()

    # Suddivisione iniziale train/test sui dati grezzi
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Preprocessing fit solo sul train e applicato al test
    X_train, y_train, preprocess_artifacts = fit_preprocess_train(
        X_train_raw,
        y_train_raw,
        class_balancer=class_balancer,
        corr=corr_threshold
    )
    X_test, y_test = transform_with_fitted_preprocess(X_test_raw, y_test_raw, preprocess_artifacts)

    remaining_feature_names = preprocess_artifacts['remaining_feature_names']
    print(remaining_feature_names)

    # Caricamento iper parametri
    best_params, best_score = load_best_params()
    if not os.path.exists(param_file_path):
        # Eseguire l'ottimizzazione bayesiana se il file non esiste
        print("Eseguendo l'ottimizzazione bayesiana...")

        best_params, best_score = bayesian_optimization(
            X_train_raw,
            y_train_raw,
            cv=k,
            class_balancer=class_balancer,
            corr=corr_threshold,
            n_iter=25,
            scorer=make_scorer(false_negative_rate, greater_is_better=False)
            # scorer=make_scorer(precision_score, greater_is_better=True)
            # scorer=make_scorer(recall_score, greater_is_better=True)
            # scorer=make_scorer(matthews_corrcoef, greater_is_better=True)
        )
        save_best_params(best_params, param_file_path)
        print("Ottimizzazione bayesiana eseguita")
    print(f"Iperparametri caricati: {best_params}")

    # Cross-validation

    k_fold_metrics, k_fold_sk_metrics = k_fold_cross_validation(
        X_train_raw,
        y_train_raw,
        ModelName,
        k=k,
        model_params=best_params,
        class_balancer=class_balancer,
        corr=corr_threshold
    )
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
        plot_learning_curve_with_kfold(
            model=LogisticRegressionGD(**best_params),
            X=X_train_raw,
            y=y_train_raw,
            cv=k,
            preprocess_fit_fn=fit_preprocess_train,
            preprocess_apply_fn=transform_with_fitted_preprocess,
            preprocess_kwargs={"class_balancer": class_balancer, "corr": corr_threshold},
            model_name=ModelName.LOGISTIC_REGRESSION_GD.value
        )
        plot_learning_curve_with_kfold(
            model=LogisticRegression(max_iter=best_params.get('n_iterations', 1000)),
            X=X_train_raw,
            y=y_train_raw,
            cv=k,
            preprocess_fit_fn=fit_preprocess_train,
            preprocess_apply_fn=transform_with_fitted_preprocess,
            preprocess_kwargs={"class_balancer": class_balancer, "corr": corr_threshold},
            model_name=ModelName.SCIKIT_LEARN.value
        )
        plot_graphs(X_train, y_train, y_test, test_predictions, test_sk_predictions, ModelName, remaining_feature_names)
        plot_results(X_test, y_test, model, sk_model, test_predictions, test_sk_predictions, scores,
                     sk_scores, ModelName)

    end_time = time.time()
    print(f"\nTempo di esecuzione totale: {end_time - start_time:.4f} secondi")
