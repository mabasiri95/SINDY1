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


################ INPUTS ###############

t = np.linspace(0, 1, 101)
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1) 




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
    feature_library=ps.PolynomialLibrary(degree=4),
    optimizer=ps.STLSQ(threshold=0.2), #fit_intercept=(True)
    feature_names=["x", "y"]
)
model_1.fit(X, t=t)
model_1.print()
print("\nCandidate Library:\n",model_1.feature_library.get_feature_names())



############### TESTING THE MODEL ##########


def plot_simulation(model, x0, y0):
    t_test = np.linspace(0, 3, 301)
    x_test = x0 * np.exp(-2 * t_test)
    y_test = y0 * np.exp(t_test)

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




