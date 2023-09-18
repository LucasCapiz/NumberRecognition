import numpy as np
 
# Inicializamos aleatoriamente los valores de los Theta entre [-Epsilon, +Epsilon]
def initialize(a, b):
  epsilon = 0.15
  c = np.random.rand(a, b + 1) * (2 * epsilon) - epsilon 
  return c
