import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры задачи
L = 10.0      # Длина области
T = 15.0       # Время моделирования
Nx = 1600      # Число узлов по x
Nt = 3200      # Число временных шагов
D = 0.1      # Коэффициент диффузии
v = 0.001       # Скорость адвекции

# Шаги сетки
dx = L / (Nx - 1)
dt = T / Nt

# Численные параметры
r = D * dt / dx**2
Pe = v * dt / (2 * dx)  # Число Пекле для адвекции

# Сетка по x
x = np.linspace(0, L, Nx)

# Начальное условие: нулевое поле
u = np.zeros(Nx)

u_new = np.copy(u)

# Матрица A (левая часть)
A = np.zeros((Nx-2, Nx-2))
for i in range(Nx-2):
    A[i, i] = 1 + 2 * r
    if i > 0:
        A[i, i-1] = -r - Pe
    if i < Nx-3:
        A[i, i+1] = -r + Pe

# Матрица B (правая часть)
B = np.zeros((Nx-2, Nx-2))
for i in range(Nx-2):
    B[i, i] = 1 - 2 * r
    if i > 0:
        B[i, i-1] = r + Pe
    if i < Nx-3:
        B[i, i+1] = r - Pe

# Источник
source_position = 2.0  # Положение источника
source_index = int(source_position / dx)  # Индекс источника
source_strength = 0.3  # Сила источника

# Список для хранения решений на каждом шаге
solutions = []

# Решение во времени
for n in range(Nt):
    # Добавляем источник
    u[source_index] += source_strength * dt

    # Формируем правую часть
    b = B @ u[1:-1]

    # Решаем СЛАУ для нового слоя
    u_inner = np.linalg.solve(A, b)

    # Обновляем решение
    u[1:-1] = u_inner

    # Краевые условия (например, нулевые Неймана)
    u[0] = u[1]
    u[-1] = u[-2]

    # Сохраняем решение для анимации
    solutions.append(u.copy())

solutions = np.array(solutions)

# Анимация
fig, ax = plt.subplots()
line, = ax.plot(x, solutions[0], label="Численное решение")
ax.set_xlim(0, L)
ax.set_ylim(0, 1.1*solutions.max())
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.legend()
ax.grid()

def update(frame):
    current_time = frame * dt
    line.set_ydata(solutions[frame])
    ax.set_title(f"Concentration at t = {current_time:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=len(solutions), interval=50)
plt.show()
