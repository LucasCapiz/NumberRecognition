from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialize import initialize
from Prediction import predict
from scipy.optimize import minimize

# Cargamos el DataSet
dataSet = loadmat('mnist-original.mat')
X = dataSet['data']
X = X.transpose()
X = X / 255 # Nos devuelve la escala de grises entre 0 y 1


# Extraemos las etiquetas
Y = dataSet['label']
Y = Y.flatten() # Flaten convierte una matriz en un vector unidimensional. Lo "aplana"

# Datos de entranamiento
X_train = X[:60000, :]
Y_train = Y[:60000]

# Datos de prueba
X_test = X[60000:, :]
Y_test = Y[60000:]

# Cantidad de datos de entrenamientos (Filas)
samples = X.shape[0]


# Definimos los parametros de la red neuronal
input_layer_size = 784 # Corresponden a cada pixel de las imagenes de 28x28 = 784
hidden_layer_size = 100
n_labels = 10


# Inicializamos los pesos (weights) de la red neuronal de manera aleatoria
initial_Theta1 = initialize(hidden_layer_size, input_layer_size)
initial_Theta2 = initialize(n_labels, hidden_layer_size)


# Inicializamos los parametros de la red neuronal
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
n_iterations = 100
lamba_value = 0.1 # Sobreajuste
myargs = (input_layer_size, hidden_layer_size, n_labels, X_train, Y_train, lamba_value)


# Utilizamos la red neuronal
results = minimize(neural_network, x0=initial_nn_params, args=myargs,
			options= {'disp': True, 'maxiter': n_iterations}, method="L-BFGS-B", jac=True)


# Extraemos los Theta
nn_params = results["x"]


# Weights are split back to Theta1, Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
							hidden_layer_size,		input_layer_size + 1)) # shape = (100, 785)
Theta2 = np.reshape(nn_params[hidden_layer_size *  (input_layer_size + 1):],
					(n_labels, hidden_layer_size + 1)) # shape = (10, 101)


# Comprobamos la efectividad del conjunto de pruebas del modelo
prediction = predict(Theta1, Theta2, X_test)
print('Training set accuracy: {:f}'.format((np.mean(prediction == Y_test) * 100)))


# Comprobamos la efectividad del conjunto de entrenamiento del modelo
prediction = predict(Theta1, Theta2, X_train)
print('Training set accuracy: {:f}'.format((np.mean(prediction == Y_train) * 100)))


# Evaluamos la precision del modelo
true_positive = 0
for i in range(len(prediction)):
    if prediction[i] == Y_train[i]:
        true_positive += 1
false_positive = len(Y_train) - true_positive
print('Precision =', true_positive/(true_positive + false_positive))


# Guardando los Thetas en un .txt
np.savetxt('Theta1.txt', Theta1, delimiter='  ')
np.savetxt('Theta2.txt', Theta2, delimiter='  ')