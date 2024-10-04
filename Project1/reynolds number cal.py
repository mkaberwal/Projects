# Calculate Re and print for viscous flow
# Lists of values for density, velocity, and viscosity
densities = [14.96e-4, 10.66e-4]  # List of densities in kg/m^3
velocities = [100, 450]  # List of velocities in m/s
viscosities = [3.430e-7, 3.217e-7]  # List of dynamic viscosities in kg/(m*s)

# Function to calculate Reynolds number
def calculate_reynolds_number(density, velocity, length, dynamic_viscosity):
    return (density * velocity * length) / dynamic_viscosity

# Loop through each combination and calculate Reynolds number
for density in densities:
    for velocity in velocities:
        for viscosity in viscosities:
            length = 1  # Assuming a characteristic length of 1 meter
            reynolds_number = calculate_reynolds_number(density, velocity, length, viscosity)
            print(f"For density={density}, velocity={velocity}, and viscosity={viscosity}, the Reynolds number is: {reynolds_number}")