import numpy as np


def predict(Theta1, Theta2, X):
	samples = X.shape[0]
	one_matrix = np.ones((samples, 1))
	X = np.append(one_matrix, X, axis=1) 
	weights_sum = np.dot(X, Theta1.transpose())
	aux_matrix = 1 / (1 + np.exp(-weights_sum)) 
	one_matrix = np.ones((samples, 1))
	aux_matrix = np.append(one_matrix, aux_matrix, axis=1)
	weights_sum2 = np.dot(aux_matrix, Theta2.transpose())
	aux_matrix3 = 1 / (1 + np.exp(-weights_sum2)) 
	prediction = (np.argmax(aux_matrix3, axis=1)) 
	
	return prediction
