"""
Created on Wed Nov 2 2016

@author: Rahul Dandwate
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('linearexpansion.csv', delimiter=',',names=['x', 'y'])

X = data['x']
Y = data['y']
N = np.shape(X)
N = N[0]

A = np.sum(X)
temp = np.square(X)
B = np.sum(temp)
C = np.sum(Y)
temp = X*Y
D = np.sum(temp)
temp = np.square(Y)
E = np.sum(temp)

def log_likelihood(a, b):
    temp = 0
    for i in range(0, N):
        temp += (Y[i] - ((a * X[i]) + b))**2
    return temp

def parameter():
    det = A**2 - (N * B)
    a = ((A * C) - (N * D)) / det
    b = ((A * D) - (C * B)) / det
    return a, b

def error(theta):                                                               #computing the error for the
    a = np.ones(N)                                                              #minimized parameters
    x = np.column_stack((X, a))
    return np.sqrt((np.sum((np.dot(x, theta) - Y)**2)) / (N - 2))

a, b = parameter()
theta = np.array([a, b])
error = error(theta)
L_max = log_likelihood(a, b)

def error_b():
    p = np.zeros(3)
    p[0] = N
    p[1] = (2 * a * A) - (2 * C)
    p[2] = E + ((a ** 2) * B) - (2 * a * D) - (error ** 2) - L_max
    error_val = np.roots(p)
    return error_val

def error_a():
    p = np.zeros(3)
    p[0] = B
    p[1] = (2 * b * A) - (2 * D)
    p[2] = E + (N * (b ** 2)) - (2 * b * C) - (error ** 2) - L_max
    error_val = np.roots(p)
    return error_val

err_int_a_min, err_int_a_max = error_a()
err_int_b_min, err_int_b_max = error_b()

print("error interval of a: ", err_int_a_min, err_int_a_max)
print("error interval of b: ", err_int_b_min, err_int_b_max)

print(a, b)
print(error)

s = np.arange(a - 10, a + 10, 0.1)
t = np.arange(b - 30, b + 30, 0.1)
x, y = np.meshgrid(s, t)
F = ((x ** 2) * B) + (N * (y ** 2)) + (E) + (2 * x * y * A) - (2 * y * C) - (2 * x * D)
G1 = (2 * (error ** 2)) + L_max
G2 = (4 * 2 * (error ** 2)) + L_max

CS1 = plt.contour(x, y, (F - G1), [0], colors = 'r')
CS2 = plt.contour(x, y, (F - G2), [0])

labels = ['1 - Sigma Interval', '2 - Sigma Interval']

CS1.collections[0].set_label(labels[0])
CS2.collections[0].set_label(labels[1])

plt.legend(loc = 'upper right')
plt.xlabel('Slope of line (m)')
plt.ylabel('Intercept of line (c)')
plt.title('Sigma Interval')
plt.show()
