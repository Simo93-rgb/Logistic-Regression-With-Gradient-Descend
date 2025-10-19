import matplotlib.cm as cm
from funzioni import *
from logistic_regression_with_gradient_descend import LogisticRegressionGD


def plot_gradient_descent(X, y, model, i=0, num_points=6, save_file=False, feature=""):
    print(f'plot feature {feature}')

    # Definire i limiti della curva della funzione di costo
    theta_values = np.linspace(-9, 9, 100)
    cost_values = []

    # Calcolo della funzione di costo per vari theta
    for theta in theta_values:
        model.theta[i] = float(theta)  # Assicurati che theta sia un valore scalare
        linear_model = np.dot(X, model.theta) + model.bias
        h = model.sigmoid(linear_model)
        epsilon = 1e-10  # Piccolo valore per evitare logaritmo di 0

        # Calcolo del costo
        cost = - (1 / len(y)) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        cost_values.append(cost)

    # Tracciare la funzione di costo
    plt.plot(theta_values, cost_values, label="Cost function curve")

    # Selezionare solo pochi punti da theta_history
    theta_history = np.array(model.theta_history)  # Ora è una matrice, ogni riga è un vettore theta
    cost_history = []

    # Calcolo dei costi corrispondenti a theta[i] durante l'addestramento
    for theta in theta_history[:, i]:  # Prendi solo i valori di theta per la feature i
        model.theta[i] = float(theta)  # Assicurati che theta[i] sia un valore scalare
        linear_model = np.dot(X, model.theta) + model.bias
        h = model.sigmoid(linear_model)
        epsilon = 1e-10  # Piccolo valore per evitare logaritmo di 0

        # Calcolo del costo
        cost = - (1 / len(y)) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        cost_history.append(cost)

    # Selezionare solo un sottoinsieme di punti
    if len(theta_history) > num_points:
        indices = np.linspace(0, len(theta_history) - 1, num=num_points, dtype=int)
        theta_history = theta_history[indices, i]  # Prendi solo i theta della feature i
        cost_history = np.array(cost_history)[indices]

    # Parametrizzazione per lo spazio di visualizzazione (usa range più ampio per un zoom più generale)
    min_cost = min(cost_history) - 0.1  # Margine più ampio per la visualizzazione del costo
    max_cost = max(cost_history) + 0.1
    min_theta = -10  # Usa un range più ampio per theta
    max_theta = 10
    plt.xlim(min_theta, max_theta)
    plt.ylim(min_cost, max_cost)

    # Creare una mappa di colori e ridurre la dimensione dei punti
    colors = cm.rainbow(np.linspace(0, 1, num_points))
    sizes = np.linspace(200, 50, num_points)  # Primo punto grande, gli altri più piccoli

    # Tracciare i punti con dimensioni e colori diversi
    labels = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
    for j, (theta, cost, color, size, label) in enumerate(zip(theta_history, cost_history, colors, sizes, labels)):
        plt.scatter(theta, cost, color=color, s=size, label=f'{label.capitalize()}')

    # Impostare la legenda
    plt.legend(title="Gradient Descent Steps")

    # Titolo e assi con il nome della feature
    plt.title(f"Gradient Descent Path for {feature}")
    plt.xlabel(f"Theta[{feature}]")
    plt.ylabel("Cost")
    plt.grid(True)

    if save_file:
        plt.savefig(f'Assets/thetas/theta_{feature}.png', format='png', dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_all_thetas(X, y, model, remaining_feature_names, num_points=6):
    num_thetas = len(model.theta)  # Numero di theta corrispondente al numero di feature
    for i in range(num_thetas):
        # Passare il nome della feature al posto dell'indice
        plot_gradient_descent(X,
                              y,
                              model,
                              i,
                              num_points,
                              feature=remaining_feature_names[i],
                              save_file=True
                              )


# Esegui il codice come prima, passando anche la lista dei nomi delle feature
X, y = carica_dati()
X_normalized, features_eliminate, y_encoded = preprocessa_dati(
    X,
    y,
    normalize=True,
    class_balancer="",
    corr=1.01,
    save_dataset=False
)
# Ottieni i nomi delle feature
all_feature_names = X.columns
remaining_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if
                           i not in features_eliminate]

# Utilizzo del metodo per plottare la curva di discesa del gradiente
model = LogisticRegressionGD(
    learning_rate=0.001,
    n_iterations=100000,
    regularization='none',
    lambda_=0.1)
model.fit(X_normalized, y_encoded)
# Esempio di utilizzo con una delle feature
# plot_theta_history([th[0] for th in model.theta_history], 'radius1')

print('Start plotting')
plot_all_thetas(X_normalized, y_encoded, model, remaining_feature_names)
# plot_gradient_descent(X_normalized, y_encoded, model, feature='smoothness2', save_file=True, i=14)