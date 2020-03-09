import numpy as np
import matplotlib.pyplot as plt
import gradientDescentMulti
import time
from mpl_toolkits import mplot3d
import featureNormalize

# Prepare the data
data = np.loadtxt('maindata2.txt', delimiter = ',')

#print(data)
x_1 = data[:,0:2]
Y = data[:,2].reshape(np.shape(x_1)[0],1)
m = np.shape(x_1)[0]


# Print some data points:
print('First 10 examples from the dataset: \n');
print(f' X = {np.around(x_1[0:10,:],2)},  Y = {np.around(Y[0:10,:],2)} \n');
time.sleep(1.5) # pause for 1.5 secs

#scale features ab d set them to zero mean
print('Normalizing Features ...\n')

x_1, mu, sigma = featureNormalize.featureNormalize(x_1)

#adding the intercept term to  feature data
X = np.ones((m,np.shape(data)[1]))
X[:,1:np.shape(data)[1]] = x_1


## Gradient Descent
print('Running gradient descent ...\n')

# Choose some alpha values
alpha = 0.01
iteration = 400

# Initial theta and run gradient descent
theta = np.zeros((np.shape(data)[1],1))

theta, J_history = gradientDescentMulti.gradientDescent(X, Y, theta, alpha, iteration)

#PLot the convergence graph
plt.figure(1)
plt.plot(J_history)
plt.grid(linestyle ='--'); 
plt.xlabel('Number of iterations');
plt.ylabel('Cost Function J'); 
plt.title('Plot of the Cost Function in house price prediction\n')
plt.show()
time.sleep(2) # pause for 2 secs
#Display gradient's descent results

print('Theta computed from gradient descent: \n');
print(f'{theta}');
print('\n');


#Estimate the price of a 1650 sq-ft, 3 br house
price =  [1, 1.650, 3] @ theta;

print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${price[0].round(2)}\n\n');
time.sleep(1.5) # pause for 1.5 secs


## PRedictions using Normal Equations!!

#preapare the data
print('Computing theta from normal equation ...\n')
phi = np.ones((m,np.shape(data)[1]))
phi[:,1:] = data[:,0:2]

theta_hat = np.linalg.inv(phi.T @ phi) @ (phi.T @ Y)
print('Theta computed using Normal Equation: \n');
print(f'{theta_hat}');
print('\n');

price1 =  [1, 1.650, 3] @ theta_hat;

print(f'Predicted price of a 1650 sq-ft, 3 br house (using Normal Equation):\n ${price1[0].round(2)}\n\n');










































