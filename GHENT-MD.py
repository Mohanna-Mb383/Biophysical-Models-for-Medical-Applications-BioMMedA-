import numpy as np
import matplotlib.pyplot as plt

# Constants
sigma = 3.4e-10  # meters
epsilon = 0.24 * 1.60218e-19  # Joules
mass = 6.63e-26  # kg (mass of Argon atom)
time_step = 1e-15  # seconds
num_steps = 10000  # Total simulation steps

# Functions
def lj_force(r):
    r6 = r ** 6
    r12 = r6 ** 2
    force_mag = 48 * epsilon * ((sigma ** 12 / r12) - 0.5 * (sigma ** 6 / r6)) / r
    return force_mag

def compute_forces(positions):
    forces = np.zeros_like(positions)
    num_particles = len(positions)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r > 1e-10:  # Prevent division by zero
                f = lj_force(r) * (r_vec / r)
                forces[i] += f
                forces[j] -= f
    return forces

def remove_com_velocity(velocities):
    com_velocity = np.mean(velocities, axis=0)
    velocities -= com_velocity
    return velocities

# Load initial positions and velocities
positions = []
velocities = []

with open("Ar_initial.txt") as file:
    lines = file.readlines()
    pos_start = lines.index("Positions (Angstrom)\n") + 1
    vel_start = lines.index("Velocities (Angstrom/picosecond)\n") + 1
    
    # Parse positions
    for line in lines[pos_start:vel_start - 2]:
        values = line.split()
        if len(values) == 3:  # Ensure the line has exactly three values
            _, x, y = map(float, values)
            positions.append([x * 1e-10, y * 1e-10])  # Convert Å to meters
    
    # Parse velocities
    for line in lines[vel_start:]:
        values = line.split()
        if len(values) == 3:  # Ensure the line has exactly three values
            _, vx, vy = map(float, values)
            velocities.append([vx * 100, vy * 100])  # Convert Å/ps to m/s

positions = np.array(positions)
velocities = np.array(velocities)
velocities = remove_com_velocity(velocities)

# Simulation
kinetic_energy = []
potential_energy = []
temperature = []

for step in range(num_steps):
    forces = compute_forces(positions)
    
    # Velocity-Verlet integration
    positions += velocities * time_step + 0.5 * forces / mass * time_step ** 2
    new_forces = compute_forces(positions)
    velocities += 0.5 * (forces + new_forces) / mass * time_step

    # Energy calculations
    ke = 0.5 * mass * np.sum(velocities ** 2)
    pe = 0.0
    num_particles = len(positions)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-10:  # Avoid division by zero
                pe += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    kinetic_energy.append(ke)
    potential_energy.append(pe)
    
    # Temperature calculation for 2D system
    temp = (2 * ke) / (num_particles * 1.38e-23 * (2 * num_particles - 2))
    temperature.append(temp)

    if step % 100 == 0:
        print(f"Step: {step}, KE: {ke:.2e} J, PE: {pe:.2e} J, Total Energy: {ke + pe:.2e} J, Temperature: {temp:.2f} K")

# Plotting results
time = np.arange(num_steps) * time_step

plt.figure()
plt.plot(time, kinetic_energy, label="Kinetic Energy")
plt.plot(time, potential_energy, label="Potential Energy")
plt.plot(time, np.array(kinetic_energy) + np.array(potential_energy), label="Total Energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.legend()
plt.title("Energy over Time")
plt.show()

plt.figure()
plt.plot(time, temperature, label="Temperature")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Temperature over Time")
plt.show()
