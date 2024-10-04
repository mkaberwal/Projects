## Imports ##
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyxfoil

N = 300
x_start, x_end = -5.0, 5.0
y_start, y_end = -1.5, 1.5

x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)
X, Y = np.meshgrid(x,y)


#### Varibles ####

a = 0
v_inf = 2
c = 3
Re = 0
alf = 0
t = .3

foil = '0030'
naca = True
pyxfoil.GetPolar(foil, naca, alf, Re)

filename = 'Data/naca0030/naca0030_surfCP_Re0.00e+00a0.0.dat'
x30 , y30, Cp = np.loadtext(filename, unpack = True, skiprows = 3)
