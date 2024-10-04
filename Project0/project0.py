
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sympy as sp
import seaborn as sns
#Get color cycle for manual colors
colors = sns.color_palette()
savedir = 'results'

#Varibles
L= 300
Re = 1e8
# Define the range for the non-dimensional height variable ynon
ynon = np.linspace(0, 1, 201) #non-dim. vertical location in BL
unon = ynon ** (1 / 7) #1/7th power law velocity distribution (turbulent BL)
ulaminar = (2*(ynon) - ynon**2)

#PROBLEM 1A: PLOT FLAT PLATE BOUNDARY LAYER VELOCITY PROFILE
#Start Figure (figszie sets aspect ratio of plot)
plt.figure(figsize=(6,6))
plt.title('Boundary Layer Velocity Profile\n(Re={:1.1E})'.format(Re)) #figure￿

plt.xlabel("$u/u_e$") #Label x axis (dollar signs are for number formatting￿

plt.ylabel("$y / \\delta$") #Label y axis
line = 2 #linewidth
#plot velocity profile
plt.plot(unon, ynon, label='Turbulent', linestyle='--', color='red')
#plot u/ue as vertical line for reference
vertlinex = np.zeros(len(ynon)) #yaxis
plt.plot(vertlinex, ynon, color=colors[0], linewidth=line) #plot zero-velocity￿
plt.plot(ulaminar, ynon, label='Laminar', linestyle='-', color='blue')

#Fill between two lines
plt.fill_betweenx(ynon, vertlinex, unon, facecolor=colors[0], alpha=0.2) #fill￿

#Plot arrows between two lines:
wd, ln = 0.03, 0.03 #arrow head dimensions
for i in range(0, len(ynon), 15):
    if abs(unon[i]) < ln:
        plt.plot([0, unon[i]], [ynon[i], ynon[i]], color=colors[0],linewidth=line)
else:
    plt.arrow(0, ynon[i], unon[i]-ln, 0, head_width=wd, head_length=ln,
        fc=colors[0], ec=colors[0], linewidth=line)
plt.axis([min(unon), max(unon), min(ynon), max(ynon)]) #limit plot bounds
plt.axis('equal') #Force equal scales on both axes ("apples to apples")
#Save Figure. File extension (i.e. '.png', '.pdf') will set filetype
plt.savefig('{}/pj0_1_BLVelProfile.png'.format(savedir), bbox_inches='tight')

#1.2 Boundray Layer Thinkness
# Define the boundary layer velocity profile u/ue from the previous calculations

#non-dim. displacement thickness
delta_star_non = np.trapz(1 - unon, ynon) #integrate vel. profile accorind to￿
#dimensional boundary layer thicknes at aft [ft]
delta = L * 0.16 / (Re ** (1 / 7)) #Blausius turbulent flat plate BL thickness￿
#re-dimensionalize disp. thickness
delta_star = delta_star_non * delta
print('|------------------------------------------------------------------|')
print('|Displacement Thickness: {} in (Re={:1.1E}, L={})|'.format(delta_star*12,Re, L) )
print('|------------------------------------------------------------------|')
print('BL Thickness: {} in'.format(delta*12) )


######################################################################################
#2.1 Airfoil plots
import matplotlib.pyplot as plot
#LOADING DATA
datadir = 'Data' #Directory to load airfoil data from
file1 = 'naca0006.dat'
file2 = 'naca4412.dat'
file3 = 'naca16018.dat'
#PLOT ALL AIRFOILS

## NACA 0006
x, y = np.loadtxt(file1, skiprows = 2, unpack = True)


## NACA 4412
n, b = np.loadtxt(file2, skiprows = 2, unpack = True)


## GOE 274
v, z = np.loadtxt(file3, skiprows = 2, unpack = True)

#plot.ylim([-0.5, 0.5])
plt.figure(figsize=(8,4))
plt.title('Airfoil Geometries') #Set title of figure
plt.xlabel("x/c") #Label x axis (non-dimensional x)
plt.ylabel("z/c") #Label y axis (non-dimensional z)
plot.plot(x,y, label='NACA006', linestyle='-', color='blue')
plot.plot(n,b, label='NACA4412', linestyle='--', color='red')
plot.plot(z,v, label='NACA16018', linestyle='-.', color='green')
plt.legend()
plot.show()

####################
import mses
#filename = "n6409_polar.dat"
#alpha, Cl, Cd, A,B,C,D = np.loadtxt(filename, skiprows = 1, unpack = True)
#alpha_desired = 0.35
#Cl_desired = np.interp(alpha_desired, alpha, Cl)
filename = "naca4412.dat"
x,z = np.loadtxt(filename, skiprows = 1, unpack = True)
upperX, lowerX = mses.MsesSplit(x, x) #split the x coordinate
upper, lower = mses.MsesSplit(x, z) #split the z coordinate
# For maximum thickness
# Print the len(upper) and print(len(lower)) and check if they have the same length
# If length is the same then:
T = upper - lower
print(max(T))
# Find the location where maximum thickness occurs:
indexT = np.where(T == max(T))
print(indexT)
locationT = upperX[indexT]
print(locationT)
# For max camber:
camberline = (upper+lower) / 2
camber = max(camberline)
print(camber)
# Find the location where maximum camber occurs:
index = np.where(camberline == camber)
print(index)
locationC = lowerX[13]
print(locationC)
plt.figure()
plt.title('Airfoil Geometry Characteristics')
plt.plot(upperX, upper, label = 'upper surface')
plt.plot(lowerX, lower, label = 'lower surface')
plt.plot(locationC, camber, marker = 'o', color = 'r')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()


############################################################################
#3.1
#LOAD NACA 2412 SURFACE PRESSURE DATA
#'naca2412_SurfPress_a6.csv' is a text file with three columns:
#x/c, lower surface Cp, and upper surface Cp
#each column is separated by commas, so we need the " delimiter=',' " option
#'unpack' option gives the values for each column to its own variable
#'skiprows=1' skips the first title row that does not contain actual data
#NOTE: you may need to adjust number of rows to skip, depending on the file
filename = 'naca2412_SurfPress_a6.csv'
x, Cpl, Cpu = np.loadtxt(filename, skiprows=1, unpack=True, delimiter=',')
plt.figure(figsize=(6,6))
plt.title('Surface Pressure Distribution \n NACA 2412 $(\\alpha=6^o)$')
plt.xlabel("x/c") #Label x axis (non-dimensional x)
plt.ylabel("$C_P$") #Label y axis Pressure coefficient
plt.gca().invert_yaxis() #MUST EITHER PLOT NEGATIVE CP OR REVERSE Y AXIS*******
#Plot Airfoil Data
plt.plot(x, Cpu, label='Upper', linewidth=2, linestyle='--')
plt.plot(x, Cpl, label='Lower', linewidth=2, linestyle='-.')
plt.grid(True) #Plot a grid
plt.xlim([0, 1]) #Lock x-axis to airfoil
plt.legend(loc='best') #Legend
#Save Figure. File extension (i.e. '.png', '.pdf') will set filetype
#3.2
dCpudx = np.zeros(len(Cpu) - 1) #forward difference so one less point
dCpldx = np.zeros(len(Cpl) - 1)
for i in range(len(dCpudx)):
    dCpudx[i] = (Cpu[i+1] - Cpu[i]) / (x[i+1] - x[i]) #slope equation
dCpldx[i] = (Cpl[i+1] - Cpl[i]) / (x[i+1] - x[i])
#Plot Pressure Gradient
plt.figure(figsize=(6,6))
plt.title('Surface Pressure Gradient \n NACA 2412 $(\\alpha=6^o)$')
plt.xlabel("x/c") #Label x axis (non-dimensional x)
plt.ylabel("$dC_P$/dx") #Label y axis Pressure coefficient
#Plot Airfoil Data
plt.plot(x[:-1], dCpudx, label='Upper', linewidth=2, linestyle='--')
plt.plot(x[:-1], dCpldx, label='Lower', linewidth=2, linestyle='-.')
plt.grid(True) #Plot a grid
plt.xlim([0, 1]) #Lock x-axis to airfoil
plt.ylim([-10, 10]) #Limit bounds to see smaller results
plt.legend(loc='best') #Legend
#Save Figure. File extension (i.e. '.png', '.pdf') will set filetype
plt.savefig('{}/pj0_3_SurfPressGrad.png'.format(savedir), bbox_inches='tight')


####################
#Linear Algerbra
w, x, y, z = sp.symbols('w x y z')
# Define the coefficient matrix A and the constant vector b
A = np.array([[4, 2, 3, 2],
              [-3, 1, -2, 3],
              [0, 1, 2, 1],
              [3, 1, -1, -2]])

b = np.array([10, 9, -3, -5])

# Solve the system of equations
solution = np.linalg.solve(A, b)

# Print the solution
print('Results in:')
for var, val in zip('wxyz', solution): #loop through letters in string and list
    print('{} = {:1.6f}'.format(var, val))


######################
# Air Foil Curve
#LOAD NACA 2412 LIFT CURVE DATA
filename = 'xf-naca1412-il-50000.csv'
a, Cl = np.loadtxt(filename, skiprows=1, unpack=True, delimiter=',')
plt.figure(figsize=(25,2))
plt.title('NACA 1412 Lift Curve')
plt.xlabel("alpha") #Label x axis (non-dimensional x)
plt.ylabel("Cl") #Label y axis Pressure coefficient
plt.plot(a, Cl, label='Upper', linewidth=2, linestyle='--')
plt.grid(True) #Plot a grid
plt.legend() #Legend
alf_target = 5.65 #make sure same units as data you're interpolating against***
Cl_target = np.interp(alf_target, a, Cl)
print('alpha={}, Cl={}'.format(alf_target, Cl_target))

### M1 airfoil
filename = 'xf-m1-il-50000.csv'
a, Cl = np.loadtxt(filename, skiprows=1, unpack=True, delimiter=',')
plt.figure(figsize=(25,2))
plt.title('NACA M1 Lift Curve')
plt.xlabel("alpha") #Label x axis (non-dimensional x)
plt.ylabel("Cl") #Label y axis Pressure coefficient
plt.plot(a, Cl, label='Upper', linewidth=2, linestyle='--')
plt.grid(True) #Plot a grid
plt.legend() #Legend
alf_target = 5.65 #make sure same units as data you're interpolating against***
Cl_target = np.interp(alf_target, a, Cl)
print('alpha={}, Cl={}'.format(alf_target, Cl_target))


