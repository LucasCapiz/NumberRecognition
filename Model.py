import numpy as np

def neural_network(nn_params, input_layer_size, hidden_layer_size, n_labels, X, Y, lamba_value):

    # Theta1 [100 x 785] --> hidden_layer_size, input_layer_size + 1
    # Theta2 [10  x 101] --> n_labels, hidden_layer_size + 1
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (n_labels, hidden_layer_size + 1))
    

    # Forward propagation
    samples = X.shape[0]
    one_matrix = np.ones((samples, 1))
    X = np.append(one_matrix, X, axis=1)
    aux_matrix = X
    weights_sum = np.dot(X, Theta1.transpose())
    aux_matrix2 = 1 / (1 + np.exp(-weights_sum)) # Sigmoide activation
    aux_matrix2 = np.append(one_matrix, aux_matrix2, axis=1)
    weights_sum2 = np.dot(aux_matrix2, Theta2.transpose())
    aux_matrix3 = 1 / (1 + np.exp(-weights_sum2)) # Sigmoide activation


    # Matriz identidad
    identity_matrix = np.zeros((samples, 10))
    for i in range(samples):
        identity_matrix[i, int(Y[i])] = 1


    # Funcion de coste
    cost_value = (1 / samples) * (np.sum(np.sum(-identity_matrix * np.log(aux_matrix3) - (1 - identity_matrix) * np.log(1 - aux_matrix3)))) + (lamba_value / (2 * samples) )
    sum(sum(pow(Theta1[:,1:], 2))) + sum(sum(pow(Theta2[:,1:], 2)))


    # Backpropagation
    Error_output_layer = aux_matrix3 - identity_matrix
    Error_hidden_layer = np.dot(Error_output_layer, Theta2) * aux_matrix2 * (1 - aux_matrix2)
    Error_hidden_layer = Error_hidden_layer[:, 1:]  

    
    # Gradiente
    Theta1[:,0] = 0 # Eliminamos los sesgos, es decir la primer columna de todas las filas
    Theta1_grad = (1 / samples) * np.dot(Error_hidden_layer.transpose(), aux_matrix) + (lamba_value / samples) * Theta1

    Theta2[:,0] = 0 # Eliminamos los sesgos, es decir la primer columna de todas las filas
    Theta2_grad = (1 / samples) * np.dot(Error_output_layer.transpose(), aux_matrix2) + (lamba_value / samples) * Theta2
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    
    return cost_value, grad