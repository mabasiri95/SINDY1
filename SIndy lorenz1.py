#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:55:52 2023

@author: mohammadaminbasiri
"""

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.integrate import odeint


################ INPUTS ###############

t = np.linspace(0, 1, 101)
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1) 

# def f(state, t):
#     x, y = state
#     return -2 * x, y

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

state0 = [1.0, 1.0, 1.0]

time_steps = np.arange(0.0, 1.0, 0.01)

x_train = odeint(f, state0, time_steps)



plt.figure(figsize=(6, 4))
#plt.plot(x, y, label="signal", linewidth=4)
plt.plot(x_train[:, 0], x_train[:, 1], "--", label="Dynamics", linewidth=3)
plt.plot(state0[0], state0[1], "ko", label="Initial condition", markersize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


#######3 another Input ##########

def f(state, t):
    x, y = state
    return -2 * x - 3 * y, -y - x

state0 = [3.0, 0.5]

time_steps = np.arange(0.0, 1.0, 0.01)

x_train = odeint(f, state0, time_steps)



plt.figure(figsize=(6, 4))
plt.plot(x_train[:, 0], x_train[:, 1], "--", label="Dynamics", linewidth=3)
plt.plot(state0[0], state0[1], "ko", label="Initial condition", markersize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

X = x_train
t = time_steps

############## TRAINING THE MODEL ########


### FOURIER 
model = ps.SINDy(
    differentiation_method=ps.FiniteDifference(order=2),
    feature_library=ps.FourierLibrary(n_frequencies=1),
    optimizer=ps.STLSQ(threshold=0.2), #intercept is for adding constant ,  fit_intercept=(True)
    feature_names=["x", "y"]
)
model.fit(X, t=t)
model.print()
print("\nCandidate Library:\n",model.feature_library.get_feature_names())

### POLY
model_1 = ps.SINDy(
    differentiation_method=ps.FiniteDifference(order=2),
    feature_library=ps.PolynomialLibrary(degree=1),
    optimizer=ps.STLSQ(threshold=0.2), #fit_intercept=(True)
    feature_names=["x", "y"]
)
model_1.fit(X, t=t)
model_1.print()
print("\nCandidate Library:\n",model_1.feature_library.get_feature_names())



############### TESTING THE MODEL ##########


def plot_simulation(model, x0, y0):
    
    state0 = [x0,y0]
    t_test = np.arange(0.0, 2.0, 0.01)
    X_test = odeint(f, state0, t_test)
    x_test = X_test[:,0]
    y_test = X_test[:,1]

    sim = model.simulate([x0, y0], t=t_test)

    plt.figure(figsize=(6, 4))
    plt.plot(x_test, y_test, label="Ground truth", linewidth=4)
    plt.plot(sim[:, 0], sim[:, 1], "--", label="SINDy estimate", linewidth=3)
    plt.plot(x0, y0, "ko", label="Initial condition", markersize=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    plt.plot(t_test, x_test, label="Ground truth", linewidth=4)
    plt.plot(t_test, sim[:, 0], "--", label="SINDy estimate", linewidth=3)
    plt.plot(0, x0, "ko", label="Initial condition", markersize=8)
    plt.xlabel("time")
    plt.ylabel("x")
    
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    plt.plot(t_test, y_test, label="Ground truth", linewidth=4)
    plt.plot(t_test, sim[:, 1], "--", label="SINDy estimate", linewidth=3)
    plt.plot(0, y0, "ko", label="Initial condition", markersize=8)
    plt.xlabel("time")
    plt.ylabel("y")
    
    
    
x0 = 3
y0 = 0.5
plot_simulation(model, x0,y0)
plot_simulation(model_1, x0,y0)

x0 = 6
y0 = -0.5
plot_simulation(model, x0,y0)
plot_simulation(model_1, x0,y0)




##### Ploting time series #####


def plot_dimension(dim, name):
    fig = plt.figure(figsize=(9,2))
    ax = fig.gca()
    ax.plot(t, x_train[:, dim])
    ax.plot(t, x_sim[:, dim], "--")
    plt.xlabel("time")
    plt.ylabel(name)

plot_dimension(0, 'x')
plot_dimension(1, 'y')





############# PLOT ############

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.plot(t, x, color='red', label='x', linewidth=3)
ax.plot(t, y, color='blue', label='y', linewidth=3)
ax.legend(loc='upper left', shadow=True)
# plt.savefig('accuracy.png')
ax.set_title('Signals')
ax.set_xlabel('Time')
ax.set_ylabel('Amp')
#ax.set_ylim(bottom=0, top = 100000)
#ax.set_xlim(left=2)
ax.grid(True)
#plt.yscale("log")
plt.show()




