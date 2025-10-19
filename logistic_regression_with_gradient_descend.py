import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionGD(BaseEstimator, ClassifierMixin):
    """Classificatore di Regressione Logistica utilizzando la Discesa del Gradiente.

       Questa classe implementa la regressione logistica per problemi di classificazione binaria
       utilizzando l'ottimizzazione tramite discesa del gradiente. Supporta la regolarizzazione
       L1 (Lasso) e L2 (Ridge).

       Attributes:
           learning_rate (float): Tasso di apprendimento per l'aggiornamento dei parametri.
           n_iterations (int): Numero massimo di iterazioni per la discesa del gradiente.
           tolerance (float): Soglia per la variazione della perdita per determinare la convergenza.
           regularization (str): Tipo di regolarizzazione ('ridge', 'lasso' o None).
           lambda_ (float): Forza della regolarizzazione.
           theta (np.ndarray): Parametri dei pesi.
           bias (float): Parametro di bias.
           losses (list): Valori della perdita per ogni iterazione.
       """

    def __init__(self, learning_rate=0.1, n_iterations=1000, tolerance=1e-6, regularization='none', lambda_=0.0):
        """Inizializza il classificatore LogisticRegressionGD.

                Args:
                    learning_rate (float, optional): Tasso di apprendimento. Default a 0.1.
                    n_iterations (int, optional): Numero massimo di iterazioni. Default a 1000.
                    tolerance (float, optional): Soglia per la convergenza. Default a 1e-10.
                    regularization (str, optional): Tipo di regolarizzazione ('ridge', 'lasso' o None). Default a 'ridge'.
                    lambda_ (float, optional): Forza della regolarizzazione. Default a 0.01.
                """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_ = lambda_
        self.theta = None
        self.bias = None
        self.losses = []
        self.classes_ = None
        self.theta_history = []  # Aggiunto per tracciare i valori di theta

    def sigmoid(self, z):
        """Calcola la funzione sigmoide.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Output dopo l'applicazione della funzione sigmoide.
        """
        z = np.clip(z, -500, 500)  # Limita z per evitare overflow in exp
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, print_iteration=False):
        """Addestra il modello sui dati forniti.

                Args:
                    X (np.ndarray): Matrice delle caratteristiche di forma (n_campioni, n_caratteristiche).
                    y (np.ndarray): Vettore target di forma (n_campioni,).
                    print_iteration (bool, optional): Se True, stampa la perdita a intervalli regolari. Default a False.

                Raises:
                    ValueError: Se il parametro 'regularization' non è valido.
                """
        self.classes_ = np.unique(y)  # Memorizza le classi uniche del target
        y = np.array(y).ravel()
        n_samples, n_features = X.shape
        # self.theta = np.zeros(n_features)
        self.theta = np.random.uniform(4.5, 4.5, size=n_features)

        self.bias = 0
        self.losses = []
        self.theta_history = []

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.theta) + self.bias
            h = self.sigmoid(linear_model)

            # Precauzione per evitare NaN
            h = np.clip(h, 1e-10, 1 - 1e-10)

            # Funzione di costo con regolarizzazione
            if self.regularization == 'ridge':
                regularization_term = (self.lambda_ / (2 * n_samples)) * np.sum(np.square(self.theta))
            elif self.regularization == 'lasso':
                regularization_term = (self.lambda_ / n_samples) * np.sum(np.abs(self.theta))
            else:
                regularization_term = 0

            loss = - (1 / n_samples) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + regularization_term
            self.losses.append(loss)
            if print_iteration and i % (self.n_iterations // 10) == 0:
                print(f'Iteration {i}, Loss: {loss}')

            # Memorizza il valore corrente di theta[0] (primo parametro)
            self.theta_history.append(np.copy(self.theta))
            # Discesa del gradiente
            dw = (1 / n_samples) * np.dot(X.T, (h - y))
            db = (1 / n_samples) * np.sum(h - y)

            # Aggiornamento con regolarizzazione
            if self.regularization == 'ridge':
                dw += (self.lambda_ / n_samples) * self.theta
            elif self.regularization == 'lasso':
                dw += (self.lambda_ / n_samples) * np.sign(self.theta)

            # # Stampa il gradiente per monitorare l'andamento
            # print(f"Iteration {i}, Gradient dw: {dw}, db: {db}")
            self.theta -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i > 0 and abs(self.losses[-2] - loss) < self.tolerance:
                print(f"Convergence reached at iteration {i}")
                print(
                    f'Params: lr={self.learning_rate}, lambda={self.lambda_}, reg={self.regularization}, iter={self.n_iterations}')
                break

    def predict(self, X):
        """Predice le etichette di classe per i campioni di input.

                Args:
                    X (np.ndarray): Matrice delle caratteristiche di forma (n_campioni, n_caratteristiche).

                Returns:
                    np.ndarray: Etichette di classe predette (0 o 1) per ciascun campione.
                """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.theta) + self.bias
        probabilities = self.sigmoid(linear_model)
        return np.vstack([1 - probabilities, probabilities]).T

    # Metodo per ottenere i parametri (necessario per scikit-learn)
    def get_params(self, deep=True):
        """Ottiene i parametri per questo stimatore.

                Args:
                    deep (bool, optional): Per compatibilità con scikit-learn. Default a True.

                Returns:
                    dict: Nomi dei parametri mappati ai loro valori.
                """
        return {
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'tolerance': self.tolerance,
            'regularization': self.regularization,
            'lambda_': self.lambda_
        }

    # Metodo per impostare i parametri (necessario per scikit-learn)
    def set_params(self, **params):
        """Imposta i parametri per questo stimatore.

                Args:
                    **params: Nomi dei parametri mappati ai nuovi valori.

                Returns:
                    LogisticRegressionGD: Se stesso con i parametri aggiornati.
                """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def plot_losses(self):
        """Traccia la curva della funzione di perdita in base alle iterazioni."""
        plt.plot(self.losses)
        plt.title('Loss Function over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.ylim(0, 0.15)
        plt.grid(True)
        plt.show()



