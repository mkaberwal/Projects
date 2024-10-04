### Project 4 ###

### Imports ###
import numpy as np
import matplotlib.pyplot as plt



############################################################


### Problem 1 - Vortex Potentiel Flow ###

# Set up grid
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Define functions for potential flow
def vortex(strength, xv, yv, X, Y):
    u = +strength/(2*np.pi) * (Y - yv) / ((X - xv)**2 + (Y - yv)**2)
    v = -strength/(2*np.pi) * (X - xv) / ((X - xv)**2 + (Y - yv)**2)
    return u, v

def sink(strength, xs, ys, X, Y):
    u = -strength/(2*np.pi) * (X - xs) / ((X - xs)**2 + (Y - ys)**2)
    v = -strength/(2*np.pi) * (Y - ys) / ((X - xs)**2 + (Y - ys)**2)
    return u, v

# Vortex
gamma = 5.0
u_vortex, v_vortex = vortex(gamma, 0, 0, X, Y)

# Vortex + Sink
lambda_ = -1
u_vortex_sink, v_vortex_sink = vortex(gamma, 0, 0, X, Y)
u_vortex_sink += sink(lambda_, 0, 0, X, Y)[0]
v_vortex_sink += sink(lambda_, 0, 0, X, Y)[1]

# Vortex Sheet
n_vortices = 12
xv_sheet = np.linspace(-1, 1, n_vortices)
gamma_sheet = np.full(n_vortices, gamma)
u_vortex_sheet, v_vortex_sheet = 0, 0
for i in range(n_vortices):
    u_i, v_i = vortex(gamma_sheet[i], xv_sheet[i], 0, X, Y)
    u_vortex_sheet += u_i
    v_vortex_sheet += v_i

# Infinite Vortex Sheet
u_inf_vortex_sheet, v_inf_vortex_sheet = vortex(gamma, 0, 0, X, Y)

# Plotting
plt.figure(figsize=(15, 10))

# Vortex
plt.subplot(2, 2, 1)
plt.streamplot(X, Y, u_vortex, v_vortex)
plt.title('Vortex')
plt.xlabel('X')
plt.ylabel('Y')

# Vortex + Sink
plt.subplot(2, 2, 2)
plt.streamplot(X, Y, u_vortex_sink, v_vortex_sink)
plt.title('Vortex + Sink')
plt.xlabel('X')
plt.ylabel('Y')

# Vortex Sheet
plt.subplot(2, 2, 3)
plt.streamplot(X, Y, u_vortex_sheet, v_vortex_sheet)
plt.title('Vortex Sheet')
plt.xlabel('X')
plt.ylabel('Y')

# Infinite Vortex Sheet
plt.subplot(2, 2, 4)
plt.streamplot(X, Y, u_inf_vortex_sheet, v_inf_vortex_sheet)
plt.title('Infinite Vortex Sheet')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()

################################################################################



### Problem 2 - Vortex Panel Method ###











#################################################################################



### Problem 3 - Calculating Lift using Circulation










#################################################################################