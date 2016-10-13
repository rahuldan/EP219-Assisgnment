"""
Created on Sun Oct 8 2016

@author: Rahul Dandwate
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('linearexpansion.csv', delimiter=',',names=['x', 'y'])     #Reading in the data
num_iter = 100000                                                               #Number of iterations for gradient descent
m = np.shape(data['x'])
gamma = 0.01                                                                    #Learning parameter for gradient descent

def gradient_descent():                                                         #Gradient descent algorithm to find
    theta = np.ones(2)                                                          #the best fit line parameters.
    a = np.ones(m)                                                              #The cost function(least square distance)
    X = np.column_stack((data['x'], a))                                         #is minimized by moving along the
    for i in range(0, num_iter):                                                #gradient of the cost function
        temp = np.zeros(2)                                                      #for a given number of iterations
        temp[0] = temp[0] + np.sum(np.multiply(np.dot(X, theta) - data['y'], data['x']))
        temp[1] = temp[1] + np.sum((np.dot(X, theta) - data['y']))
        theta[0] = theta[0] - (gamma * temp[0] / m[0])
        theta[1] = theta[1] - (gamma * temp[1] / m[0])
    return theta

def error(theta):                                                               #computing the error for the
    a = np.ones(m)                                                              #minimized parameters
    X = np.column_stack((data['x'], a))
    return np.sqrt((np.sum((np.dot(X, theta) - data['y'])**2)) / (m[0] - 2))

theta = gradient_descent()
error1 = error(theta)
x = np.linspace(0, 10)
print('Coefficient of linear expansion = ', theta[0], ' mm/Kelvin')
print('Expected Length at 0 Degree C = ', theta[1], ' mm')
print('Expected Length at 15 Degree C = ', (theta[0]*15) + theta[1], ' mm')
print('Error in Length at 15 Degree C = ', error1, ' mm')
plt.plot(x, (x*theta[0]) + theta[1])                                            #plotting the line
plt.title('Best Fit Line')
plt.scatter(data['x'], data['y'], alpha = 0.5)                                  #plotting the data
plt.xlabel('Temperature (Degree Celsius)')
plt.ylabel('Length (mm)')
plt.show()
