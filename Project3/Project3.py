import math
import pyxfoil
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class Panel:
    
    def __init__(self, xa, ya, xb, yb):
        
        self.xa, self.ya = xa, ya           # Defines the first end point
        self.xb, self.yb = xb, yb           # Defines the second end point

        # Defining center point and panel parameters
        # You will need to define these yourself:
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2         # Control point or center point (How do you find the center of two points?)
        self.length = math.sqrt((xb - xa)**2 + (yb - ya)**2)    # Length of the panel (How do you find the distance between two points)

        # For the orientation of the panel (angle between x axis and the unit vector normal to the panel)
        if xb - xa <= 0:
            self.beta = math.acos((yb - ya) / self.length)
        elif xb - xa > 0:
            self.beta = math.pi + math.acos(-(yb - ya) / self.length)

        # Location of the panel (we will use this later when we expand our analys to airfoils)
        if self.beta <= math.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'

    
        # Create these and set the equal to zero for now

        self.sigma = 0.0 
        self.vt = 0.0
        self.cp = 0.0
        

def integral_normal(p_i, p_j):
    """
    Evaluates the contribution of a panel at the center-point of another,
    in the normal direction.

    Parameters:
    -----------
    p_i: Panel object
        Panel on which the contribution is calculated.
    p_j: Panel object
        Panel from which the contribution is calculated. 
    """

    def integrand(s):
        return (((p_i.xc - (p_j.xa - math.sin(p_j.beta) * s)) * math.cos(p_i.beta) +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s)) * math.sin(p_i.beta)) /
                ((p_i.xc - (p_j.xa - math.sin(p_j.beta) * s))**2 +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s))**2))
    return integrate.quad(integrand, 0.0, p_j.length)[0]
    
def integral_tangential(p_i, p_j):

    def integrand(s):
        return ((-(p_i.xc - (p_j.xa - math.sin(p_j.beta) * s)) * math.sin(p_i.beta) +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s)) * math.cos(p_i.beta)) /
                ((p_i.xc - (p_j.xa - math.sin(p_j.beta) * s))**2 +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s))**2))
    return integrate.quad(integrand, 0.0, p_j.length)[0]

def integral_tangential(p_i, p_j):

    def integrand(s):
        return ((-(p_i.xc - (p_j.xa - math.sin(p_j.beta) * s)) * math.sin(p_i.beta) +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s)) * math.cos(p_i.beta)) /
                ((p_i.xc - (p_j.xa - math.sin(p_j.beta) * s))**2 +
                 (p_i.yc - (p_j.ya + math.cos(p_j.beta) * s))**2))
    return integrate.quad(integrand, 0.0, p_j.length)[0]

def analyze_panels(panels, u_inf):
    
    Num = len(panels)

    # First we need the Normal Velocity Calculations

    A_n = np.empty((Num, Num), dtype = float)
    np.fill_diagonal(A_n, 0.5)
        # Whenever we have i = j, we have sigma(i)/2 or sigma(i)*0.5. Thus, on our diagonal for matrix A we should have 0.5 
        # The diagonal of a matrix means i = j i.e (1,1), (2,2), etc etc.

    # Create the source influence matrix [A] of the linear system
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A_n[i,j] = (0.5/math.pi) * integral_normal(p_i, p_j)

    # Create the right hand side [b] of the linear system
    b_n = - u_inf * np.cos([p.beta for p in panels])

    sigma = np.linalg.solve(A_n,b_n)

    for i, panel in enumerate(panels):
        panel.sigma = sigma[i]

    # ====================================================================== #
    # Now we need the Tangential Velocity Calculations
    A_t = np.empty((Num, Num), dtype = float)
    np.fill_diagonal(A_t, 0.0)

    # Create the source influence matrix [A] of the linear system
    # STUDENTS WILL FILL THIS IN DELETE LATER
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i!=j:
                A_t[i,j] = (0.5 / math.pi) * integral_tangential(p_i, p_j)

    # Create the right hand side [b] of the linear system
    # STUDENTS ALSO FILL THIS IN
    b_t = -u_inf * np.sin([p.beta for p in panels])

    # Finally, we compute the tangential velocity:
    # STUDENTS ALSO FILL THIS IN
    vt = np.dot(A_t, sigma) + b_t

    for i, panel in enumerate(panels):
        panel.vt = vt[i]
    
    # STUDENTS FILL THIS IN
    # Finally, lets use our tangential velocity to calculate surface pressure Cp:
    for panel in panels:
        panel.cp = 1.0 - (panel.vt/u_inf)**2

    print('Panel Analysis Complete!')

N_panels = np.array([10, 40, 160])
R = 0.75

x_circle = R * np.cos(np.linspace(0.0, 2 * math.pi, 100))
y_circle = R * np.sin(np.linspace(0.0, 2 * math.pi, 100))

# Update the for loop to iterate over the length of the array N_panels
cylinder_panels = np.empty(len(N_panels), dtype=object)
for j in range(len(N_panels)):
    x_points = R * np.cos(np.linspace(0.0, 2 * math.pi, N_panels[j] + 1))
    y_points = R * np.sin(np.linspace(0.0, 2 * math.pi, N_panels[j] + 1))

    # Create panel objects and store them in the array cylinder_panels
    for i in range(N_panels[j]):
        cylinder_panels[j] = Panel(x_points[i], y_points[i], x_points[i + 1], y_points[i + 1])
def plot_figure(N_panels, R, x_circle, y_circle, x_points, y_points, cylinder_panels):
    plt.figure(figsize=(7, 7))
    plt.grid()
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Number of panels: %d' % N_panels, fontsize=16)
    plt.plot(x_circle, y_circle, label='circle/cylinder', color='b', linewidth=1)
    plt.plot(x_points, y_points, label='panels', color='r')

    plt.scatter([p.xa for p in cylinder_panels], [p.ya for p in cylinder_panels], label='end points', color='r', linewidth=2)
    plt.scatter([p.xc for p in cylinder_panels], [p.yc for p in cylinder_panels], label='center points', color='k', linewidth=2)

    plt.legend(loc='best')
    plt.show()

# Call the function for each value in the N_panels array
N_panels = np.array([10, 40, 160])
R = 0.75
x_circle = R * np.cos(np.linspace(0.0, 2 * math.pi, 100))
y_circle = R * np.sin(np.linspace(0.0, 2 * math.pi, 100))

for panels in N_panels:
    x_points = R * np.cos(np.linspace(0.0, 2 * math.pi, panels + 1))
    y_points = R * np.sin(np.linspace(0.0, 2 * math.pi, panels + 1))
    
    cylinder_panels = np.empty(panels, dtype=object)
    for i in range(panels):
        cylinder_panels[i] = Panel(x_points[i], y_points[i], x_points[i + 1], y_points[i + 1])

    plot_figure(panels, R, x_circle, y_circle, x_points, y_points, cylinder_panels)

def calculate_error(cylinder_panels, x_circle, cp_analytical, N_panels):
    d_theta = 2 * math.pi / N_panels
    integral_analytical = np.sum(cp_analytical * d_theta)
    integral_panel = np.sum([p.cp * d_theta for p in cylinder_panels])

    error = integral_analytical - integral_panel
    return error

def analyze_and_plot_panels(cylinder_panels, U_infty, x_circle, y_circle, cp_analytical, N_panels):
    analyze_panels(cylinder_panels, U_infty)
    cp_analytical = 1.0 - 4 * (y_circle / R) ** 2

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.xlabel('X', fontsize=16)
    plt.ylabel('$C_p$', fontsize=16)

    plt.plot(x_circle, cp_analytical, label='analytical', color='b', linestyle='-', linewidth=1, zorder=1)

    plt.scatter([p.xc for p in cylinder_panels], [p.cp for p in cylinder_panels], label='source-panel method', color='#CD2305', s=40, zorder=2)

    plt.title('Number of panels: %d' % N_panels, fontsize=16)
    plt.legend(loc='best', prop={'size': 16})
    plt.xlim(-1.5, 1.5)
    plt.ylim(-4.0, 2.0)
    plt.show()

# Call the function for each value in the N_panels array
N_panels = np.array([10, 40, 160])
R = 0.75
U_infty = 1.0
x_circle = R * np.cos(np.linspace(0.0, 2 * math.pi, 100))
y_circle = R * np.sin(np.linspace(0.0, 2 * math.pi, 100))
cp_analytical = 1.0 - 4 * (y_circle / R)**2

for panels in N_panels:
    x_points = R * np.cos(np.linspace(0.0, 2 * math.pi, panels + 1))
    y_points = R * np.sin(np.linspace(0.0, 2 * math.pi, panels + 1))
    
    cylinder_panels = np.empty(panels, dtype=object)
    for i in range(panels):
        cylinder_panels[i] = Panel(x_points[i], y_points[i], x_points[i + 1], y_points[i + 1])

    analyze_and_plot_panels(cylinder_panels, U_infty, x_circle, y_circle, cp_analytical, panels)

    error = calculate_error(cylinder_panels, x_circle, cp_analytical, panels)
    print(f"Error for {panels} panels: {error}")

def define_panels(x, y, N= 100):
   
    R = (x.max() - x.min()) / 2                 # Radius of the circle, based on airfoil geometry
    x_center = (x.max() + x.min()) / 2          # X coordinate of center of circle


    x_circle = x_center + R * np.cos(np.linspace(0.0, 2 * math.pi, N + 1))
    # Here we define the x coordinates of the circle
    
    x_ends = np.copy(x_circle)                  # projection of the x-coord on the surface
    y_ends = np.empty_like(x_ends)              # initialization of the y-coord Numpy array

    x, y = np.append(x, x[0]), np.append(y, y[0])  # extend arrays using numpy.append
    
    # computes the y-coordinate of end-points
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]
    
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])
    
    return panels

### NACA 0012 ###
naca = True
alf = 0.0
foil = '0012'
kinvisc = 15.52 * (10**-6)
Re = (10*1)/(kinvisc)
Re = 0
pyxfoil.GetPolar(foil, naca, alf, Re)

filename = 'Data/naca0012/naca0012.dat'
x, y = np.loadtxt(filename, dtype=float, unpack=True, skiprows = 1)

N = 100
panels = define_panels(x,y,N)

# Plot the geometry and the panels
width = 10
plt.figure(figsize=(width, width))
plt.grid()
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)
plt.plot(np.append([panel.xa for panel in panels], panels[0].xa),
            np.append([panel.ya for panel in panels], panels[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
plt.axis('equal')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.1)
plt.show()

# Analysis using xfoil
Re_values = [0, 2.5108e+5]  # example Reynolds numbers
plt.figure(figsize=(10, 6))
plt.grid()
plt.xlabel('x', fontsize=16)
plt.ylabel('$C_p$', fontsize=16)

for Re in Re_values:
    pyxfoil.GetPolar(foil, naca, alf, Re)
    filename = f'Data/naca0012/naca0012_surfCP_Re{Re:.2e}a{alf}.dat'
    x12, y12, Cp12 = np.loadtxt(filename, unpack=True, skiprows=3)

    U_infty = 5
    analyze_panels(panels, U_infty)

    plt.plot(x12, Cp12, label=f'Re = {Re}')

plt.gca().invert_yaxis()
plt.plot([p.xc for p in panels], [p.cp for p in panels], label='source-panel method', color='#CD2305')
plt.title('Xfoil vs Source Panel', fontsize=16)
plt.legend(loc='best', prop={'size': 16})
plt.show()

naca = True
alf = 0.0
foil = '2414'
kinvisc = 15.52 * (10**-6)
Re = (10*1)/(kinvisc)
Re = 0
pyxfoil.GetPolar(foil, naca, alf, Re)

filename = 'Data/naca2414/naca2414.dat'
x, y = np.loadtxt(filename, dtype=float, unpack=True, skiprows = 1)

N = 100
panels = define_panels(x,y,N)

# Plot the geometry and the panels
width = 10
plt.figure(figsize=(width, width))
plt.grid()
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)
plt.plot(np.append([panel.xa for panel in panels], panels[0].xa),
            np.append([panel.ya for panel in panels], panels[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
plt.axis('equal')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.1)
plt.show()

# Analysis using xfoil
Re_values = [0, 2.5108e+5]  # example Reynolds numbers
plt.figure(figsize=(10, 6))
plt.grid()
plt.xlabel('x', fontsize=16)
plt.ylabel('$C_p$', fontsize=16)

for Re in Re_values:
    pyxfoil.GetPolar(foil, naca, alf, Re)
    filename = f'Data/naca2414/naca2414_surfCP_Re{Re:.2e}a{alf}.dat'
    x12, y12, Cp12 = np.loadtxt(filename, unpack=True, skiprows=3)

    U_infty = 5
    analyze_panels(panels, U_infty)

    plt.plot(x12, Cp12, label=f'Re = {Re}')

plt.gca().invert_yaxis()
plt.plot([p.xc for p in panels], [p.cp for p in panels], label='source-panel method', color='#CD2305')
plt.title('Xfoil vs Source Panel', fontsize=16)
plt.legend(loc='best', prop={'size': 16})
plt.show()