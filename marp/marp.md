---
marp: true
theme: neobeam
size: 16:9
paginate: true
math: mathjax
style: |
  :root {
    --font-size: 44px;
    --header-height: 2.2rem;
    --text-padding: 0.45em;
  }
  section {
    justify-content: flex-start;
  }
  section.title {
    justify-content: center;
  }

footer: '
  **Simone Garau**
  **Logistic Regression with Gradient Descent**
  **Machine Learning, 2026**'
---

<!-- _class: title -->
# Logistic Regression With Gradient Descent
## Classificazione Binaria sul Dataset Wisconsin Breast Cancer

> ### Progetto accademico in Machine Learning
> Confronto Custom GD vs Scikit-learn

## Versione relazione + presentazione

---
<!-- header: 'Indice' -->

1. Obiettivi e dataset
2. Pipeline leak-safe
3. Modello custom e scelte implementative
4. Tuning bayesiano e ablation
5. Risultati quantitativi
6. Confronto grafico e conclusioni

---
<!-- header: 'Obiettivi' -->

## Obiettivi del progetto

* Implementare da zero Logistic Regression binaria con Gradient Descent.
* Confrontare in modo corretto con la baseline scikit-learn.
* Mantenere validazione rigorosa: holdout finale + cross-validation leak-safe.
* Analizzare performance, generalizzazione e robustezza.

---
<!-- header: 'Dataset' -->

## Il dataset

* Fonte: UCI Wisconsin Breast Cancer (id = 17, via ucimlrepo).
* Task: classificazione Benigno vs Maligno.
* Feature: variabili numeriche continue.
* Metriche: Accuracy, F1, Recall, Precision, MCC, FNR/FPR, ROC AUC, PRC AUC.

---
<!-- header: 'Cenni teorici' -->

## Logistic Regression e loss

$$
p(y=1\mid x) = \sigma(z), \quad z = w^T x + b
$$

$$
\mathcal{L}(w,b) = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right] + \Omega(w)
$$

Aggiornamento con Gradient Descent:

$$
w \leftarrow w - \eta \nabla_w \mathcal{L}, \qquad b \leftarrow b - \eta \nabla_b \mathcal{L}
$$

---
<!-- header: 'Pipeline leak-safe' -->

## Protocollo anti-data leakage

1. Split train/test sui dati grezzi.
2. Preprocessing fittato solo sul train.
3. Applicazione degli artefatti al test.
4. Cross-validation sul train grezzo con preprocessing rifittato in ogni fold.

Questo evita che statistiche del test influenzino training e tuning.

---
<!-- header: 'Scelte implementative' -->

## Inizializzazione dei pesi (modello custom)

* Pesi iniziali piccoli, casuali e centrati in zero.
* random_state fissato per riproducibilita.

Perche questa scelta:

* Evita saturazione iniziale della sigmoide.
* Rende il gradiente piu informativo nelle prime iterazioni.
* Riduce variabilita tra run e rende i confronti piu robusti.

---
<!-- header: 'Tuning Bayesiano' -->

## Tuning degli iperparametri

Ricerca con ottimizzazione bayesiana (gp_minimize).

Migliori parametri del modello custom:

* Learning rate: ~0.0078
* Lambda: ~0.0002
* Iterazioni: 5133
* Regolarizzazione: Ridge

---
<!-- header: 'Ablation' -->

## Ablation: target di ottimizzazione

Confronto tra tuning su FNR e MCC:

* Parametri ottimali diversi tra i due target.
* In cross-validation MCC migliora leggermente alcune metriche aggregate.
* Sul test holdout, differenze molto piccole.

> #### Messaggio chiave
> Il comportamento del progetto resta stabile rispetto al target di tuning.

---
<!-- header: 'Risultati test' -->

## Risultati: test holdout

| Metrica | Custom (GD) | Scikit-learn |
|---|---:|---:|
| Accuracy | 0.9766 | 0.9766 |
| ROC AUC | 0.9991 | 0.9964 |
| F1 Score | 0.9677 | 0.9677 |
| Precision | 1.0000 | 1.0000 |
| Recall | 0.9375 | 0.9375 |

La versione custom e pienamente competitiva con la baseline industriale.

---
<!-- header: 'Learning Curve - Custom' -->

## Learning curve: modello custom

![bg fit right:50%](../Assets/FNR/learning_curve_LogisticRegressionGD.png)

**Lettura del grafico:**

* Training error basso e stabile.
* CV error in riduzione all'aumentare dei campioni.
* Gap residuo limitato nella parte finale.

---
<!-- header: 'Learning Curve - Baseline' -->

## Learning curve: baseline scikit-learn

![bg fit right:50%](../Assets/FNR/learning_curve_Scikit_learn.png)

**Lettura del grafico:**

* Andamento regolare e robusto.
* Gap train/CV contenuto.
* Prestazioni finali allineate al modello custom.

---
<!-- header: 'ROC e PRC' -->

## Curve Precision-Recall

<div style="display: flex; justify-content: space-between; gap: 20px;">
  <img src="../Assets/FNR/prc_auc_modello_LogisticRegressionGD.png" width="48%">
  <img src="../Assets/FNR/prc_auc_modello_Scikit_learn.png" width="48%">
</div>

Entrambi i modelli mostrano elevata capacita discriminativa.

---
## Effetto della regolarizzazione

<div style="display: flex; justify-content: space-between; gap: 20px;">
  <img src="../Assets/FNR/regularization_effect_ridge.png" width="48%">
  <img src="../Assets/FNR/regularization_effect_lasso.png" width="48%">
</div>

Trade-off tra stabilita dei coefficienti e sparsita.

---
<!-- header: 'Conclusioni' -->

## Conclusioni

> #### Esito del progetto
> Pipeline leak-safe + tuning corretto: il modello custom raggiunge risultati comparabili a scikit-learn.

* Confronto equo e riproducibile.
* Performance finali sostanzialmente equivalenti.
* Alto valore didattico grazie alla trasparenza del modello custom.

## Grazie

Domande?
