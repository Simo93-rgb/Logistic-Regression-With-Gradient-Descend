---
marp: true
theme: default
size: 16:8
class: 
  - lead
paginate: true
backgroundColor: #ffffff
math: mathjax
---

# Logistic Regression With Gradient Descent
## Classificazione Binaria sul Dataset Wisconsin Breast Cancer

---

## Obiettivi del Progetto

* **Implementazione Custom:** Sviluppare un classificatore logistico binario basato su Gradient Descent.
* **Benchmarking:** Confrontare le prestazioni con una baseline consolidata (scikit-learn).
* **Rigore Metodologico:** Utilizzare un protocollo train/test holdout e cross-validation leak-safe.
* **Analisi Approfondita:** Valutare metriche e grafici per discutere bias, varianza ed errori clinicamente rilevanti.

---

## Cenni Teorici: Il Modello

Il modello stima la probabilità di classe tramite la funzione sigmoide:

$$
p(y=1\mid x) = \sigma(z), \quad z = w^T x + b
$$

La funzione obiettivo (Log-Loss) penalizza gli errori di classificazione, con l'aggiunta di un termine di regolarizzazione per evitare l'overfitting:

$$
\mathcal{L}(w,b) = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right] + \Omega(w)
$$

Aggiornamento con Gradient Descent:

$$
w \leftarrow w - \eta \nabla_w \mathcal{L}, \qquad b \leftarrow b - \eta \nabla_b \mathcal{L}
$$

---

## Il Dataset

* **Fonte:** UCI Wisconsin Breast Cancer (id=17, via ucimlrepo).
* **Task:** Classificazione Benigno vs Maligno.
* **Features:** Variabili numeriche continue.

---

## Protocollo Anti-Data Leakage

Per garantire stime realistiche, la pipeline adotta un flusso rigorosamente leak-safe:

1. Split train/test sui dati grezzi.
2. Il preprocessing (imputazione NaN, normalizzazione Z-score) viene fittato esclusivamente sul train.
3. Applicazione delle trasformazioni al set di test.
4. Cross-validation sul train grezzo con preprocessing rifittato in ogni fold.

Questo impedisce alle statistiche del test di influenzare il training.

---

## Inizializzazione dei Pesi (Modello Custom)

Scelta implementativa:

* Pesi iniziali piccoli, casuali e centrati in zero.
* `random_state` fissato per riproducibilita.

Perche:

* Evita saturazione iniziale della sigmoide.
* Rende il gradiente piu informativo nelle prime iterazioni.
* Riduce variabilita tra run e rende i confronti piu robusti.

---

## Tuning degli Iperparametri

La ricerca usa l'ottimizzazione bayesiana (gp_minimize). 
I migliori iperparametri trovati per il modello custom sono:

* **Learning Rate:** ~0.0078
* **Lambda:** ~0.0002
* **Iterazioni:** 5133
* **Regolarizzazione:** Ridge

---

## Ablation: Target di Ottimizzazione

Confronto tra tuning su **FNR** e **MCC**:

* Parametri ottimali diversi tra i due target.
* In cross-validation, MCC mostra un miglioramento lieve su alcune metriche aggregate.
* Sul test holdout, differenze minime: i risultati di classificazione restano quasi sovrapponibili.

Messaggio principale:

* Il comportamento del progetto e stabile rispetto al target di tuning.
* La scelta finale del target va guidata dal costo applicativo.

---

## Risultati: Test Holdout

Le performance sui dati non visti mostrano un sostanziale pareggio tra i due modelli:

| Metrica | Custom (GD) | Scikit-learn |
|---|---:|---:|
| **Accuracy** | 0.9766 | 0.9766 |
| **ROC AUC** | 0.9991 | 0.9964 |
| **F1 Score** | 0.9677 | 0.9677 |
| **Precision** | 1.0000 | 1.0000 |
| **Recall** | 0.9375 | 0.9375 |

La versione custom è pienamente competitiva rispetto alla baseline industriale.

---

## Analisi Grafica: Learning Curves

![bg fit right:50%](../Assets/FNR/learning_curve_LogisticRegressionGD.png)

**Custom (GD):**

* Errore di training basso e stabile.
* Errore CV in convergenza con piu dati.

Confronto dell'apprendimento al variare del numero di campioni.

---

## Analisi Grafica: Learning Curves (Baseline)

![bg fit right:50%](../Assets/FNR/learning_curve_Scikit_learn.png)

**Scikit-learn:**

* Andamento regolare e robusto.
* Gap train/CV contenuto nella parte finale.

Risultati finali allineati al modello custom.

---

## Analisi Grafica: Curve ROC e PRC

![w:500](../Assets/FNR/prc_auc_modello_LogisticRegressionGD.png) ![w:500](../Assets/FNR/prc_auc_modello_Scikit_learn.png)

Entrambi i modelli mostrano ottima capacita discriminativa.

---

## Analisi Grafica: Confronto Metriche

![w:1250](../Assets/FNR/metrics_comparison.png)

Confronto aggregato delle metriche principali tra modello custom e baseline.

---

## Analisi Grafica: Regolarizzazione

![w:500](../Assets/FNR/regularization_effect_ridge.png) ![w:500](../Assets/FNR/regularization_effect_lasso.png)

Effetto di Ridge e Lasso sui coefficienti: trade-off tra stabilita e sparsita.

---

## Confronto e Conclusioni

* **Prestazioni:** I due modelli sono sostanzialmente equivalenti sul test holdout.
* **Robustezza:** Scikit-learn resta un benchmark forte per stabilità numerica.
* **Trasparenza:** Il modello custom consente piena ispezione di loss e traiettoria dei parametri.
* **Risultato:** Una corretta pipeline leak-safe permette a un'implementazione custom di allinearsi agli standard industriali.