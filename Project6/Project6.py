### Imports ####
import numpy as np
import math
import pyxfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D


# Given values
W = 12000  # Loaded Weight in lbs
S = 235    # Wing Area in ft^2
c = 8.48   # Root Chord Length in ft
b = 45     # Wing Span in ft
V_cruise = 586.667  # Cruise Speed in ft/s
h = 22000  # Geometric Cruise Altitude in ft
mu = 3.25e-7  # Dynamic Viscosity in slug/ft·s

# Constants for standard atmosphere model
rho = 11.866e-4  # Sea level air density in slug/ft^3

g = 32.174         # Acceleration due to gravity in ft/s^2



# Step 2: Calculate Design Reynolds Number (Re)
Re = (rho * V_cruise * c) / mu

# Step 3: Calculate CL_cruise
CL_cruise = (W / (0.5 * rho * (V_cruise**2) * S))

# Step 4: Calculate CL_2g
CL_2g = (W*2) / (0.5 * rho * (V_cruise**2) * S)

# Print the results
print(f"Design Reynolds number (Re): {Re:.2f}")
print(f"3-D lift coefficient for equilibrium, level flight (CL_cruise): {CL_cruise:.4f}")
print(f"3-D lift coefficient for equilibrium 2g turn (CL_2g): {CL_2g:.4f}")
# Airfoil data
filenames = ['Data/naca0012/naca0012.dat', 'Data/naca23012/naca23012.dat', 'Data/p51d/p51d.dat']
naca = False
alf = 0

plt.figure(figsize=(20, 12))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)

for filename in filenames:
    pyxfoil.GetPolar(filename, naca, alf, Re)
    x, z = np.loadtxt(filename, dtype=float, unpack=True, skiprows=1)
    plt.plot(x*8.48, z*8.48, label=filename)

plt.axis('equal')
plt.legend()
plt.show()
foil = '0012'
naca = True
alf = np.arange(-25.5, 25.5, 0.5)

pyxfoil.GetPolar(foil, naca, alf, Re, SaveCP=False, quiet=True, overwrite= True)
foil = '23012'
naca = True
alf = np.arange(-25.5, 25.5, 0.5)

pyxfoil.GetPolar(foil, naca, alf, Re, SaveCP=False, quiet=True, overwrite= True)
foil = 'Data/p51d/p51d.dat'
naca = False
alf = np.arange(-25.5, 25.5, 0.5)

pyxfoil.GetPolar(foil, naca, alf, Re, SaveCP=False, quiet=True, overwrite= True)
# List of filenames
filenames = [
    'Data/naca0012/naca0012_polar_Re1.82e+07a-25.5-25.0.dat',
    'Data/naca23012/naca23012_polar_Re1.82e+07a-25.5-25.0.dat',
    'Data/p51d/p51d_polar_Re1.82e+07a-25.5-25.0.dat'
]

# Initialize the plot
plt.figure(figsize=(20, 12))
plt.grid(True)
plt.xlabel('Alpha (°)', fontsize=16)
plt.ylabel('CL', fontsize=16)


# Loop through each file and plot the data
for filename in filenames:
    # Load data from file
    data = np.loadtxt(filename, dtype=float, unpack=True, skiprows=12)
    alpha, CL = data[0], data[1]  # Assuming alpha is in column 0 and CL in column 1

    # Plot the data
    plt.plot(alpha, CL, label=filename.split('/')[-1])  # Label with the file name

# Set up legend and show plot
plt.legend()
plt.show()
# Initialize the plot
plt.figure(figsize=(20, 12))
plt.grid(True)
plt.xlabel('Cl', fontsize=16)
plt.ylabel('CD', fontsize=16)
plt.xlim(-2,2)
plt.ylim(0, 0.025)

# Loop through each file and plot the data
for filename in filenames:
    # Load data from file
    data = np.loadtxt(filename, dtype=float, unpack=True, skiprows=12)
    CL, CD = data[1], data[2]  # Assuming alpha is in column 0 and CL in column 1

    # Plot the data
    plt.plot(CL, CD, label=filename.split('/')[-1])  # Label with the file name

# Set up legend and show plot
plt.legend()
plt.show()
plt.figure(figsize=(20, 12))
plt.grid(True)
plt.xlabel('Cl', fontsize=16)
plt.ylabel('CD', fontsize=16)

# Loop through each file and plot the data
for filename in filenames:
    # Load data from file
    data = np.loadtxt(filename, dtype=float, unpack=True, skiprows=12)
    CL, CD = data[1], data[2]  # Assuming alpha is in column 0 and CL in column 1

    # Plot the data
    plt.plot(CL, CD, label=filename.split('/')[-1])  # Label with the file name

# Set up legend and show plot
plt.legend()
plt.show()
# Initialize the plot
plt.figure(figsize=(20, 12))
plt.grid(True)
plt.xlabel('Alpha (°)', fontsize=16)
plt.ylabel('L/D', fontsize=16)


# Loop through each file and plot the data
for filename in filenames:
    # Load data from file
    data = np.loadtxt(filename, dtype=float, unpack=True, skiprows=12)
    alpha, CL, CD = data[0], data[1], data[2] # Assuming alpha is in column 0 and CL in column 1
    D = (1/2) * rho * (V_cruise**2) * CD * S
    L = 120000
    L_D = L/D
    # Plot the data
    plt.plot(alpha, L_D, label=filename.split('/')[-1])  # Label with the file name

# Set up legend and show plot
plt.legend()
plt.show()
# File paths
filenames = [
    'Data/naca0012/naca0012_polar_Re1.82e+07a-25.5-25.0.dat',
    'Data/naca23012/naca23012_polar_Re1.82e+07a-25.5-25.0.dat',
    'Data/p51d/p51d_polar_Re1.82e+07a-25.5-25.0.dat'
]

# CL values for interpolation
given_CL_values = [0.2501, 0.5001]  # Example values, replace with your own

# Process each file
for filename in filenames:
    # Read data from the file
    alpha, CL, CD, CDp, CM, Top, Bot = np.loadtxt(filename, skiprows=12, unpack=True)

    # Create interpolation functions
    interp_alpha = interp1d(CL, alpha, kind='linear', bounds_error=False)
    interp_CD = interp1d(CL, CD, kind='linear', bounds_error=False)

    # File header
    print(f"{'File:':<15} {filename.split('/')[-1]}\n")
    print(f"{'Condition':<15} {'Alpha':<10} {'Cl':<10} {'D':<10} {'L/D':<10}")

   # Interpolate and print values for each CL value
    for given_CL in given_CL_values:
        alpha_interpolated, CD_interpolated = interp_alpha(given_CL), interp_CD(given_CL)
        D = (1/2) * rho * (V_cruise**2) * CD_interpolated * S

        if given_CL == 0.2501:  # Check if given_CL is 0.2501 for level flight
            L = 12000  # example value in Newtons for Lift Force
            L_D = L / D
        # Level Flight condition
            print(f"{'Level Flight':<15} {alpha_interpolated:<10.3f} {given_CL:<10.3f} {D:<10.3f} {L_D:<10.3f}")
        elif given_CL == 0.5001:  # Check if given_CL is 0.5001 for 2G condition
            # 2G Turn condition, Lift is doubled
            L_2g = 2 * L
            L_D_2g = L_2g / D
            print(f"{'2G Turn':<15} {alpha_interpolated:<10.3f} {given_CL:<10.3f} {D:<10.3f} {L_D_2g:<10.3f}")


    print("\n")  # Just for a line break between files
given_alpha_values = [-12.5]  ## Personl pick

for filename in filenames:
    # Load the data
    data = np.loadtxt(filename, skiprows=12)
    alpha, CL, CD = data[:, 0], data[:, 1], data[:, 2]  # Assuming the first column is alpha and the second is CL
    
    # Create interpolation functions
    interp_CL = interp1d(alpha, CL, kind='linear', bounds_error=False)
    interp_CDive = interp1d(alpha, CD, kind='linear', bounds_error=False)

    # File header
    print(f"{'File:':<15} {filename.split('/')[-1]}\n")
    print(f"{'Condition':<15} {'Alpha':<10} {'Cl':<10} {'D':<10} {'L/D':<10}")

   # Interpolate and print values for each CL value
    for given_alpha in given_alpha_values:
        CL_interpolated, CDive_interpolated = interp_CL(given_alpha), interp_CDive(given_alpha)
        Dive = (1/2) * rho * (V_cruise**2) * CDive_interpolated * S
        L = 12000  # example value in Newtons for Lift Force
        L_Dive = L / Dive
        # Level Flight condition
        print(f"{'Dive':<15} {alpha_interpolated:<10.3f} {given_CL:<10.3f} {D:<10.3f} {L_Dive:<10.3f}")
    # Find the index of the maximum CL value
    max_index = np.argmax(CL)
    
    # Find the corresponding alpha value
    CLmax = CL[max_index]
    alphamax = alpha[max_index]
    CDmax = CD[max_index]
    Dmax = (1/2)*(rho)*((V_cruise)**2)*(CDmax)*S
    L = 12000  # example value in Newtons for Lift Force
    L_Dmax = L / Dmax
    
    print(f"{'Stall':<15} {alphamax:<10.3f} {CLmax:<10.3f} {Dmax:<10.3f} {L_Dmax:<10.3f}")

    print("\n")
filenames = ['Data/PanelData/Panel_1_Coordinates.dat', 'Data/PanelData/Panel_2_Coordinates.dat', 'Data/PanelData/Panel_3_Coordinates.dat', 'Data/PanelData/Panel_4_Coordinates.dat', 'Data/PanelData/Planform_Coordinates.dat']

plt.figure(figsize=(20, 12))
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)

for filename in filenames:
    # Load the data
    data = np.loadtxt(filename, skiprows=2)
    x, y = data[:, 0], data[:, 1]
    plt.plot(x, y, label = 'Panel', linestyle = '-')

controlpoints = 'Data/PanelData/Control_Points.dat'
xC, yC  = np.loadtxt(controlpoints, dtype=float, unpack=True, skiprows=2)
plt.scatter(xC, yC, label = 'Control Points', marker = 'o')

vortex = 'Data/PanelData/Vortex_Points.dat'
Xv, Yv = x, z = np.loadtxt(vortex, dtype=float, unpack=True, skiprows=2)
plt.scatter(Xv, Yv, label = 'Vortex Points', marker= 'X')

plt.axis('equal')
plt.legend()
plt.show()
filenames = ['Data/PanelData/Panel_1_Coordinates.dat', 'Data/PanelData/Panel_2_Coordinates.dat', 'Data/PanelData/Panel_3_Coordinates.dat', 'Data/PanelData/Panel_4_Coordinates.dat']

# Create a single 3D subplot
fig = plt.figure(figsize=(30, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot each panel's data
for filename in filenames:
    data = np.loadtxt(filename, skiprows=2)
    x, y = data[:, 0], data[:, 1]
    ax.plot(x, y, zs=0, zdir='z', label=filename.split('/')[-1])

# Load and plot the planform data
planform = 'Data/PanelData/Planform_Coordinates.dat'
wingplanx, wingplany = np.loadtxt(planform, dtype=float, unpack=True, skiprows=2)
ax.plot(wingplanx, wingplany, zs=0, zdir='z', label='Planform')
ax.scatter(Xv, Yv, 0) ##VOrtex Points
ax.scatter(xC, yC, 0) ## Control Points

# Show legend and plot title
ax.legend()
ax.set_title('3D Projection of P51D Wing')

# Display the plot
plt.show()
