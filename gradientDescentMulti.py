import numpy as np
import costFunction


def gradientDescent(X, Y, theta, alpha, iterations):
    """
    %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    m = len(Y)
    J_history = np.zeros((iterations,1))
    
    for i in range(iterations):
        h_theta = X @ theta
        theta = theta - ((alpha/m)* X.T @ (h_theta - Y))
        J = costFunction.costFunction(X, Y, theta)
        J_history[i] =costFunction.costFunction(X, Y, theta) 
        
    return theta, J_history