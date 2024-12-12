import numpy as np
import time

def f(t):
    return t

# Задаём параметры задачи
n_steps = 1000  # Количество шагов
t = np.linspace(0, 10, n_steps)
true_position = np.array([f(ti) for ti in t])  # Истинное положение объекта
measurement_noise_std = 1.1  # Стандартное отклонение шума измерений
process_noise_std = 0.5  # Стандартное отклонение шума системы

# Генерируем данные
np.random.seed(100)

n_steps1 = int(n_steps/2)
n_steps2 = n_steps - n_steps1

noise1 = np.random.normal(0, measurement_noise_std, size=n_steps1)
noise2 = np.random.normal(0, measurement_noise_std + 1.0, size=n_steps2)
noise = np.concatenate([noise1, noise2])

measurements = true_position + noise

# Инициализация
x_hat = np.zeros(n_steps)  # Оценка состояния
x_hat[0] = measurements[0]  # Начальная оценка
P = 1.0  # Начальная ковариация оценки
Q_k = 1000  # Инициализация Q_k (ковариация шума системы)
R_k = 1000.0  # Инициализация R_k (ковариация шума измерений)
Phi = 1.0  # Модель динамики
H = 1.0  # Модель измерений
alpha = 0.05


# Массивы для хранения адаптивных Q_k и R_k
Q_adaptive = []
R_adaptive = []

# Рекурсивный процесс фильтрации
for k in range(1, n_steps):
    # Экстраполяция
    x_pred = Phi * x_hat[k - 1]  # Предсказание состояния
    P_pred = Phi * P * Phi + Q_k  # Предсказание ковариации

    # Инновация
    innovation = measurements[k] - H * x_pred
    S = H * P_pred * H + R_k  # Ковариация инновации

    # Матрица Калмана
    K = P_pred * H / S

    # Обновление состояния
    x_hat[k] = x_pred + K * innovation
    P = (1 - K * H) * P_pred

    # Адаптивное обновление Q_k и R_k
    Q_k = alpha * (innovation ** 2) + (1.0-alpha) * Q_k  # Упрощённое обновление Q_k
    R_k = alpha * S + (1.0-alpha) * R_k  # Упрощённое обновление R_k

    Q_adaptive.append(Q_k)
    R_adaptive.append(R_k)


relative_errors = np.abs(true_position - x_hat)/np.abs(x_hat)
print(Q_k)
print(R_k)
# Визуализация
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(true_position, label="Истинное положение")
plt.plot(measurements, label="Шумные измерения", alpha=0.5)
plt.plot(x_hat, label="Оценённое положение (фильтр)", linestyle="--")
plt.legend()
plt.xlabel("Шаг")
plt.ylabel("Положение")
plt.title("Адаптивная фильтрация с обновляемыми Q_k и R_k")
plt.show()


plt.figure(figsize=(12, 6))
plt.title("Error")
plt.plot(relative_errors, label = "Ошибка истенного и оцененного")
plt.xlabel("Шаг")
plt.ylabel("Error")
plt.legend()
plt.show()

