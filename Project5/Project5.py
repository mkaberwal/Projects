### Imports ###
import math
import vivi
import pyxfoil
import numpy as np
import matplotlib.pyplot as plt
import mses
from tabulate import tabulate
from scipy import integrate

### PLOTTING DEFAULTS BOILERPLATE (OPTIONAL) #########################
#SET DEFAULT FIGURE APPERANCE
import seaborn as sns #Fancy plotting package
#No Background fill, legend font scale, frame on legend
sns.set(style='whitegrid', font_scale=1.5, rc={'legend.frameon': True})
#Mark ticks with border on all four sides (overrides 'whitegrid')
sns.set_style('ticks')
#ticks point in
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
#fix invisible marker bug
sns.set_context(rc={'lines.markeredgewidth': 0.1})
#restore default matplotlib colormap
mplcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
sns.set_palette(mplcolors)
#Get color cycle for manual colors
colors = sns.color_palette() 
#SET MATPLOTLIB DEFAULTS
    #(call after seaborn, which changes some defaults)
params = {
        #FONT SIZES
        'axes.labelsize' : 30, #Axis Labels
        'axes.titlesize' : 30, #Title
        'font.size'      : 28, #Textbox
        'xtick.labelsize': 22, #Axis tick labels
        'ytick.labelsize': 22, #Axis tick labels
        'legend.fontsize': 24, #Legend font size
        'font.family'    : 'serif',
        'font.fantasy'   : 'xkcd',
        'font.sans-serif': 'Helvetica',
        'font.monospace' : 'Courier',
        #AXIS PROPERTIES
        'axes.titlepad'  : 2*6.0, #title spacing from axis
        'axes.grid'      : True,  #grid on plot
        'figure.figsize' : (6,6),   #square plots
        'savefig.bbox'   : 'tight', #reduce whitespace in saved figures
        #LEGEND PROPERTIES
        'legend.framealpha'     : 0.5,
        'legend.fancybox'       : True,
        'legend.frameon'        : True,
        'legend.numpoints'      : 1,
        'legend.scatterpoints'  : 1,
        'legend.borderpad'      : 0.1,
        'legend.borderaxespad'  : 0.1,
        'legend.handletextpad'  : 0.2,
        'legend.handlelength'   : 1.0,
        'legend.labelspacing'   : 0,
}
import matplotlib
matplotlib.rcParams.update(params) #update matplotlib defaults, call after seaborn
### END OF BOILERPLATE ##################################################

import numpy as np
import matplotlib.pyplot as plt

# Constants
ue_freestream = 600  # Freestream velocity in kts

# Function to calculate laminar boundary layer thickness
def delta_laminar(x, Re):
    return 5.0 * x / np.sqrt(Re)

# Function to calculate turbulent boundary layer thickness
def delta_turbulent(x, Re):
    # Avoid division by zero by checking if Re is non-zero
    if Re != 0:
        return 0.16 * x / (Re**0.17)
    else:
        return 0

# Function to calculate laminar velocity profile
def u_laminar(y, delta, ue):
    return ue * ((2 * y / delta) - (y**2 / delta**2))

# Function to calculate turbulent velocity profile
def u_turbulent(y, delta, ue):
    return ue * (y / delta)**(1/7)

# Flight conditions
x_cruise = 35000  # Cruise condition
x_taxi = 0       # Taxi condition

# Calculate laminar and turbulent boundary layer thickness
Re_cruise = 600 * x_cruise / sea_level_viscosity  # Assuming sea level conditions
Re_taxi = 12 * x_taxi / sea_level_viscosity  # Assuming sea level conditions
delta_cruise_lam = delta_laminar(x_cruise, Re_cruise)
delta_taxi_lam = delta_laminar(x_taxi, Re_taxi)
delta_cruise_turb = delta_turbulent(x_cruise, Re_cruise)
delta_taxi_turb = delta_turbulent(x_taxi, Re_taxi)

# Create dimensional height values
y_values = np.linspace(0, 1, 100)

# Calculate dimensional velocity profiles
u_values_cruise_lam = u_laminar(y_values * delta_cruise_lam, delta_cruise_lam, ue_freestream)
u_values_cruise_turb = u_turbulent(y_values * delta_cruise_turb, delta_cruise_turb, ue_freestream)
u_values_taxi_lam = u_laminar(y_values * delta_taxi_lam, delta_taxi_lam, ue_freestream)
u_values_taxi_turb = u_turbulent(y_values * delta_taxi_turb, delta_taxi_turb, ue_freestream)

# Create plots
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Plot dimensional velocity profiles
axs[0].plot(u_values_cruise_lam, y_values, label='Cruise Laminar', linestyle='-', color='blue')
axs[0].plot(u_values_cruise_turb, y_values, label='Cruise Turbulent', linestyle='--', color='blue')
axs[0].plot(u_values_taxi_lam, y_values, label='Taxi Laminar', linestyle='-', color='red')
axs[0].plot(u_values_taxi_turb, y_values, label='Taxi Turbulent', linestyle='--', color='red')
axs[0].set_xlabel('Dimensional Velocity (kts)')
axs[0].set_ylabel('Dimensional Height')
axs[0].set_title('Dimensional Velocity vs. Dimensional Height')
axs[0].legend()

# Plot non-dimensional velocity profiles
axs[1].plot(u_values_cruise_lam / ue_freestream, y_values * delta_cruise_lam, label='Cruise Laminar', linestyle='-', color='blue')
axs[1].plot(u_values_cruise_turb / ue_freestream, y_values * delta_cruise_turb, label='Cruise Turbulent', linestyle='--', color='blue')
axs[1].plot(u_values_taxi_lam / ue_freestream, y_values * delta_taxi_lam, label='Taxi Laminar', linestyle='-', color='red')
axs[1].plot(u_values_taxi_turb / ue_freestream, y_values * delta_taxi_turb, label='Taxi Turbulent', linestyle='--', color='red')
axs[1].set_xlabel('Non-Dimensional Velocity (u/ue)')
axs[1].set_ylabel('Non-Dimensional Height (y/delta)')
axs[1].set_title('Non-Dimensional Velocity vs. Non-Dimensional Height')
axs[1].legend()

plt.tight_layout()
plt.show()

# Constants
uc_freestream = 308.667 
ul_freestream = 6.17333
antenna_length = 10 / 12  # Antenna length in feet (converted from inches)

# Function to calculate drag coefficient
def calculate_drag_coefficient(Re):
    Cf_lam = 1.328 / np.sqrt(Re)
    Cf_turb = 0.074 / Re**(1/5)
    return Cf_lam, Cf_turb

# Function to calculate friction drag force
def calculate_friction_drag_force(Cf, rho, A, V):
    return 0.5 * Cf * rho * A * V**2

# Flight conditions
Re_cruise = 6146.41  
Re_taxi = 997921.03

# Calculate drag coefficients
Cf_cruise_lam, Cf_cruise_turb = calculate_drag_coefficient(Re_cruise)
Cf_taxi_lam, Cf_taxi_turb = calculate_drag_coefficient(Re_taxi)

# Calculate friction drag forces (one side of the antenna fin)
A_one_side = antenna_length * antenna_width  # Area of one side of the antenna fin


Fd_cruise_turb = calculate_friction_drag_force(Cf_cruise_turb, sea_level_density, A_one_side, uc_freestream)

Fd_taxi_lam = calculate_friction_drag_force(Cf_taxi_lam, sea_level_density, A_one_side, 6.17333)  # Taxi velocity is 0


# Double the forces for both sides of the antenna fin

Fd_total_cruise_turb = 2 * Fd_cruise_turb
Fd_total_taxi_lam = 2 * Fd_taxi_lam


# Output the results
print(f"Friction drag coefficient (Cruise, Turbulent): {Cf_cruise_turb:.2f} N")
print(f"Friction drag force on the entire antenna (Cruise, Turbulent): {Fd_total_cruise_turb:.2f} N")
print(f"Friction drag coefficien (Taxi, Laminar): {Fd_total_taxi_lam:.2f} N")
print(f"Friction drag force on the entire antenna (Taxi, Laminar): {Cf_taxi_lam:.2f} N")
foil = '23012'
naca = True

# Our reynolds number doesn't matter here because we just need the geometry data.
Re = 0
alf = 0

pyxfoil.GetPolar(foil, naca, alf, Re)
filename = 'Data/naca23012/naca23012.dat'

x, z = np.loadtxt(filename, unpack = True, skiprows = 1)


plt.figure(figsize = (20,12))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.plot(x, z)
plt.axis('equal')


foil = '23012'
naca = True

Re = 5*(10**5)
alf = 12

pyxfoil.GetPolar(foil, naca, alf, Re)

filename = 'Data/naca23012/naca23012_surfCP_Re5.00e+05a12.0.dat'
x, y, Cp = np.loadtxt(filename, unpack = True, skiprows = 3)


x_upper, x_lower = mses.MsesSplit(x, x)
y_upper, y_lower = mses.MsesSplit(x, y)
Cp_up, Cp_lo = mses.MsesSplit(x,Cp)



xnew = np.linspace(0, 1, 300)

y_lower = np.interp(xnew, x_lower, y_lower)
y_upper = np.interp(xnew, x_upper, y_upper)


plt.plot(xnew, y_lower)
plt.plot(xnew, y_upper)
plt.show()


# Given values 
Re_T = 500000  # Transition Reynolds number
rho = 0.1948
V = 8
mu = 1.422E-5
xnew = np.linspace(0, 1, 300)  

Re_x = (rho * V * xnew) / mu  

# Make sure that Re_x is an array of 300 points
n = len(Re_x)

# Initialize the array to hold boundary layer thicknesses
delta = np.zeros(n)

# Set up a loop to calculate boundary layer thickness based on Reynolds number
for i in range(1, n):
    if Re_x[i] < Re_T:
       
        delta[i] = 5.0 * xnew[i] / np.sqrt(Re_x[i])  
    else:
        
        delta[i] = 0.16 * xnew[i] / (Re_x[i]**(1/7))  
        


y_lower2=y_lower-delta
y_upper2=y_upper+delta
plt.figure(figsize = (17, 9))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.axis('equal')
plt.plot(xnew, y_lower)
plt.plot(xnew, y_upper)

xnewlam, y_upperlam = xnew[1:40], y_upper[1:40]
xnewtur, y_uppertur = xnew[42:300], y_upper[42:300]
plt.plot(xnewlam, y_upperlam, color='green', label="Laminar Portion", zorder=1)
plt.plot(xnew[41], y_upper[41], marker='o', color='red', label="Transition Point")
plt.plot(xnewtur, y_uppertur, color='purple', label="Turbulent Portion")
plt.legend()
plt.plot(xnew, y_lower2, color='lightskyblue')
plt.plot(xnew, y_upper2, color='steelblue')
plt.title('NACA 23012 Airfoil Boundary Layer Thickness', fontsize=16)
plt.show()
foil = '23012'
naca = True

# Define Reynolds numbers and alpha values
Re_values = [4e4, 5e5, 6e8,]
alpha_values = [2, 8, 12]

# Iterate over alpha values
for alpha in alpha_values:
    # Create subplots for each alpha value
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

    # Iterate over Reynolds numbers
    for i, Re in enumerate(Re_values):
        # Generate polar data
        pyxfoil.GetPolar(foil, naca, alpha, Re)
        filename = f'Data/naca23012/naca23012_surfCP_Re{Re:.2e}a{alpha:.1f}.dat'
        x, y, Cp = np.loadtxt(filename, unpack=True, skiprows=3)

        xup, xlo = mses.MsesSplit(x, x)
        Cp_up, Cp_lo = mses.MsesSplit(x, Cp)

        # Plot on each subplot
        axs[i].plot(xlo, Cp_lo, marker='o', linestyle='-.', color='r', label=f'Lower Surface, Re={Re:.2e}')
        axs[i].plot(xup, Cp_up, marker='o', linestyle='-.', color='b', label=f'Upper Surface, Re={Re:.2e}')
        axs[i].invert_yaxis()
        axs[i].legend()
        axs[i].set_xlabel('Chordwise Position')

    # Set common ylabel and title for the entire figure
    axs[0].set_ylabel('Pressure Coefficient')
    plt.suptitle(f'Pressure Distribution for alpha={alpha}')


    # Show the entire figure
    plt.show()

    name = 'naca23012'
wid = 20
fig = plt.figure(figsize=(wid, wid / 4))  # save figure object as variable
ax = fig.add_subplot(1, 1, 1)  # use axis object to modify figure
ax.set_title('VIvI Geometry Iterations')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.axis('equal')

# Figure for Viscous Decambering
fig_decamber = plt.figure(figsize=(wid, wid / 4))
ax_decamber = fig_decamber.add_subplot(1, 1, 1)
ax_decamber.set_title('Viscous Decambering of Upper Surface')
ax_decamber.set_xlabel('x', fontsize=16)
ax_decamber.set_ylabel('z', fontsize=16)

# Set colors for different iterations
colors = ['k', 'r', 'y', 'g']

# Iterate over VIvI iterations
for niter in range(4):
    solution = vivi.VIvI(name, 0, niter=niter, Vinf=2, mu=1.79E-5, rho=1.225)

    # Plot upper and lower surfaces
    ax.plot(solution['up']['x'], solution['up']['z'], color=colors[niter], linestyle='--', label = f'VIvi = {niter}')
    ax.plot(solution['lo']['x'], solution['lo']['z'], color=colors[niter], linestyle='--')

    # Plot camber line
    ax.plot(solution['lo']['x'], (solution['up']['z'] + solution['lo']['z']) / 2, color=colors[niter], linestyle='--')

    # Plot viscous decambering
    ax_decamber.plot(solution['lo']['x'], (solution['up']['z'] + solution['lo']['z']) / 2, color=colors[niter], linestyle='-',label = f'VIvi = {niter}')
    


# Display legend
plt.legend()
plt.show()

# Calcualting Coefficient of Lift
filename = 'Data/naca23012_3/naca23012_3.dat'
filename2 = 'Data/naca23012/naca23012.dat'

inviscid = pyxfoil.GetPolar(filename, False, 0, Re = 0, SaveCP = True)
viscos = pyxfoil.GetPolar(filename, False, 0, Re = 1240734, SaveCP = True)

inviscid2 = pyxfoil.GetPolar(filename2, False, 0, Re = 0, SaveCP = True)
viscos2 = pyxfoil.GetPolar(filename2, False, 0, Re = 1240734, SaveCP = True)
c = 12

solutionPoints = solution['up'].copy()
solutionPoints = solutionPoints[solutionPoints.delta_star != 0]
Cd = 1/c * np.trapz(solutionPoints['tau'], solutionPoints['x'] )


solutionPoints3 = solution3['up'].copy()
solutionPoints3 = solutionPoints3[solutionPoints3.delta_star != 0]
Cd2 = 1/c * np.trapz(solutionPoints3['tau'], solutionPoints3['x'] )

data=[['Inv. Original', 0.1377, Cd],
      ['Visc. Original', 0.1239, 0],
      ['Inv. Third It.', 0.0892, Cd2],
      ['Visc. Third It.', 0.0875, 0]]

print(tabulate(data, headers=['Method', 'Cl', 'Cd']))