"""
Created on Sun Oct 8 2016

@author: Rahul Dandwate
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('linearexpansion.csv', delimiter=',',names=['x', 'y'])
num_iter = 100000
m = np.shape(data['x'])
gamma = 0.01

def gradient_descent():
    theta = np.ones(2)
    a = np.ones(m)
    X = np.column_stack((data['x'], a))
    for i in range(0, num_iter):
        temp = np.zeros(2)
        temp[0] = temp[0] + np.sum(np.multiply(np.dot(X, theta) - data['y'], data['x']))
        temp[1] = temp[1] + np.sum((np.dot(X, theta) - data['y']))
        theta[0] = theta[0] - (gamma * temp[0] / m[0])
        theta[1] = theta[1] - (gamma * temp[1] / m[0])
    return theta

def error():
    y_mean = np.sum(data['y']) / m[0]
    return np.sqrt(np.sum(((data['y'] - y_mean)**2) / (m[0] - 1)))

theta = gradient_descent()
error1 = error()
x = np.linspace(0, 10)
print('Coefficient of linear expansion = ', theta[0], ' mm/Kelvin')
print('Expected Length at 0 Degree C = ', theta[1], ' mm')
print('Expected Length at 15 Degree C = ', (theta[0]*15) + theta[1], ' mm')
print('Error in Length at 15 Degree C = ', error1, ' mm')
plt.plot(x, (x*theta[0]) + theta[1])
plt.title('Best Fit Line')
plt.scatter(data['x'], data['y'], alpha = 0.5)
plt.xlabel('Temperature (Degree Celsius)')
plt.ylabel('Length (mm)')
plt.show()
