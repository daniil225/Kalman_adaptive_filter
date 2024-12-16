import numpy as np
import matplotlib.pyplot as plt

# Define parameters
D = 0.01  # Diffusion coefficient
v = 0.1   # Advection velocity
L = 10.0  # Length of the domain
T = 5.0   # Total simulation time

nx = 100   # Number of spatial points
nt = 500   # Number of time steps

dx = L / (nx - 1)  # Spatial step
dt = T / nt        # Time step

x = np.linspace(0, L, nx)  # Spatial grid

# Source location and strength
source_position = int(0.2 * nx)  # Source at 20% of the domain
source_strength = 1.0

# Initial concentration
c = np.zeros(nx)

# Discretized matrices for advection-diffusion equation
alpha = D * dt / dx**2
beta = v * dt / (2 * dx)

# Construct the transition matrix F
F = np.zeros((nx, nx))
for i in range(1, nx - 1):
    F[i, i - 1] = alpha - beta
    F[i, i] = 1 - 2 * alpha
    F[i, i + 1] = alpha + beta

# Apply boundary conditions (Neumann: zero flux at boundaries)
F[0, 0] = 1
F[0, 1] = 0
F[-1, -2] = 0
F[-1, -1] = 1

print(F)

# Source term
G = np.zeros(nx)
G[source_position] = source_strength
print(G)

# Initialize concentration over time
concentration_history = []

for t in range(nt):
    c = F @ c + G * dt
    concentration_history.append(c.copy())

# Convert to numpy array for easier slicing
concentration_history = np.array(concentration_history)

# Plot results
plt.figure(figsize=(10, 6))

time_points = [0, int(nt * 0.25), int(nt * 0.5), int(nt * 0.75), nt - 1]
for t in time_points:
    plt.plot(x, concentration_history[t], label=f"t = {t * dt:.2f}")

plt.title("Concentration evolution in the channel")
plt.xlabel("Position (x)")
plt.ylabel("Concentration")
plt.legend()
plt.grid()
plt.show()

# Animation of the results
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, concentration_history[0])
ax.set_xlim(0, L)
ax.set_ylim(0, 1.1 * concentration_history.max())
ax.set_title("Concentration evolution over time")
ax.set_xlabel("Position (x)")
ax.set_ylabel("Concentration")

def update(frame):
    line.set_ydata(concentration_history[frame])
    ax.set_title(f"Concentration at t = {frame * dt:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=nt, interval=50)
plt.show()
