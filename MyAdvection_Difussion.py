import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def Simulation(x_start, x_end, T_start, T_end, nx, nt, D, v, noiseSigma = 0.0):
    # Derived parameters
    dx = (x_end - x_start) / nx
    dt = (T_end - T_start) / nt
    a = D * dt / (2 * dx**2)
    b = v * dt / (4 * dx)

    # Spatial and time grids
    x = np.linspace(x_start, x_end, nx + 1)
    t = np.linspace(T_start, T_end, nt + 1)

    # Initial and boundary conditions
    C0 = 1.0  # Dirichlet condition at x = 0
    C_init = np.zeros(nx + 1)  # Initial condition (e.g., zero everywhere)

    # Matrices for Crank-Nicolson
    A = np.zeros((nx + 1, nx + 1))
    B = np.zeros((nx + 1, nx + 1))

    # Fill matrices
    for i in range(1, nx):
        A[i, i - 1] = -(a - b)
        A[i, i] = 1 + 2 * a
        A[i, i + 1] = -(a + b)
        B[i, i - 1] = a - b
        B[i, i] = 1 - 2 * a
        B[i, i + 1] = a + b

    # Boundary conditions
    A[0, 0] = 1  # Dirichlet at x = 0
    A[-1, -2] = -1  # Neumann at x = L (gradient = 0)
    A[-1, -1] = 1

    B[0, 0] = 1
    B[-1, -2] = 0
    B[-1, -1] = 0

    # Time stepping
    C = C_init.copy()
    C[0] = C0  # Apply Dirichlet condition
    C_init[0] = C0
    solution = [C.copy()]

    for n in range(nt):
        noise = np.random.normal(loc = 0.0, scale=noiseSigma, size = nx+1)
        # Right-hand side
        b = B @ C
        b[0] = C0  # Dirichlet condition
        b[-1] = 0  # Neumann condition

        # Solve the linear system
        C_new = np.linalg.solve(A, b)
        C = C_new
        solution.append(C.copy() + noise)

    measurments = np.array(solution)
    return measurments, x, dt, np.linalg.inv(A) @ B, C_init


def AnimateDraw(x, x_end,dt,measurments, gif_filename = "animation.gif"):
    # Анимация
    fig, ax = plt.subplots()
    line, = ax.plot(x, measurments[0], label="Численное решение")
    ax.set_xlim(0, x_end)
    ax.set_ylim(0, 1.1*measurments.max())
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()
    ax.grid()

    def update(frame):
        current_time = frame * dt
        line.set_ydata(measurments[frame])
        ax.set_title(f"Concentration at t = {current_time:.2f}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(measurments), interval=50)
    # Сохранение анимации в GIF
    ani.save(gif_filename, writer='pillow', fps=20)  # Можно также использовать writer='pillow'

    plt.show()


def DrawSlice(data, idx):
    current_time = idx * dt
    fig, ax = plt.subplots()
    line, = ax.plot(x, data[idx], label="Численное решение")
    ax.set_xlim(0, x_end)
    ax.set_ylim(0, 1.1*data.max())
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"Concentration at t = {current_time:.2f}")
    ax.legend()
    ax.grid()
    plt.show()

x_start, x_end = 0, 10  # Spatial domain
T_start, T_end = 0, 20  # Time domain
D = 0.1  # Diffusion coefficient
v = 0.5   # Velocity
nx = 200   # Number of spatial points
nt = 300   # Number of time steps
noiseSigma = 0.05
measurments, x, dt, F, C_init = Simulation(x_start, x_end, T_start, T_end, nx, nt, D, v, noiseSigma)
prediction_x, x, dt, F, C_init = Simulation(x_start, x_end, T_start, T_end, nx, nt, D, v, 0.0)


H = np.eye(nx + 1)  # Observation matrix (identity)
Q = 100 * np.eye(nx + 1)  # Process noise covariance
R = 3.05 * np.eye(nx + 1)  # Measurement noise covariance
P = 0.001*np.eye(nx + 1)  # Initial error covariance

filter_state = np.zeros((nt, len(measurments[0])))
adaptive_QR = True

alpha = 0.99

for i in range(0, len(measurments)-1):
    z = measurments[i] 

    #predict
    x_k = prediction_x[i] #F @ x_k
    
    P = F @ P @ F.T + Q

    y = z - H @ x_k

    PHT = P @ H.T
    S = H @ PHT + R
    SI = np.linalg.inv(S)

    K = PHT @ SI
    x_k = x_k + K @ y
    
    if adaptive_QR:
        eps = z - H @ x_k
        R = alpha*R + (1-alpha)*(eps*eps.T + H @ P @ H.T)

    I_KH = np.eye(nx + 1) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    
    if adaptive_QR:
        TMP = y * y.T
        Q_prev = Q.copy()
        Q = (1-alpha)*Q + (alpha)*(K @ TMP @ K.T)
        if Q.max() > 1e-7:
            Q = Q_prev

    filter_state[i] = x_k


idx = 150
DrawSlice(measurments, idx)
DrawSlice(filter_state, idx)

#AnimateDraw(x, x_end, dt, measurments)
#AnimateDraw(x, x_end, dt, filter_state)



# for el in np.linalg.inv(A):
#     for e in el:
#         print("{:5.2f} ".format(e), end = "")
#     print()



