import numpy as np
import matplotlib.pyplot as plt

class AdaptiveKalmanFilter:
    def __init__(self, state_dim, obs_dim, lambda_rls=0.7):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.lambda_rls = lambda_rls

        # Инициализация переменных фильтра Калмана
        self.x = np.zeros((state_dim, 1))  # Состояние
        self.P = np.eye(state_dim)  # Ковариация состояния
        self.Q = np.eye(state_dim) * 1e-5  # Начальная ковариация процесса
        self.R = np.eye(obs_dim) * 1e-2  # Начальная ковариация измерений

    def predict(self, A, B, u):
        """
        Шаг предсказания фильтра Калмана
        """
        self.x = A @ self.x + B @ u
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z, H):
        """
        Шаг обновления фильтра Калмана
        """
        # Вычисление Калмановского коэффициента
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Обновление оценки состояния
        e = z - H @ self.x  # Ошибка наблюдения
        self.x = self.x + K @ e

        # Обновление ковариации состояния
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        # RLS обновление ковариаций
        self.Q = self.lambda_rls * self.Q + (1 - self.lambda_rls) * (e @ e.T)
        self.R = self.lambda_rls * self.R + (1 - self.lambda_rls) * (z - H @ self.x) @ (z - H @ self.x).T


    def run(self, A, B, H, u, z):
        """
        Полный шаг предсказания и обновления
        """
        self.predict(A, B, u)
        self.update(z, H)
        return self.x

# Основной код
if __name__ == "__main__":
    # Инициализация фильтра для 2D состояния и 2D наблюдений
    state_dim = 2
    obs_dim = 2
    kf = AdaptiveKalmanFilter(state_dim, obs_dim)

    # Генерация тестовых данных
    A = np.eye(2)  # Система без изменений
    B = np.zeros((2, 2))  # Без управления
    H = np.eye(2)  # Полное наблюдение
    u = np.zeros((2, 1))  # Нулевое управление

    # Генерация сигналов
    np.random.seed(42)
    true_states = []
    measurements = []
    estimated_states = []

    true_state = np.array([[0], [0]])
    for t in range(100):

        true_state = A @ true_state + np.random.multivariate_normal([0, 0], np.eye(2) * 0.1).reshape(-1, 1)
        z = H @ true_state + np.random.multivariate_normal([0, 0], np.eye(2) * 0.01).reshape(-1, 1)
        
        measurements.append(z)
        true_states.append(true_state)

        # Применение адаптивного фильтра Калмана
        estimate = kf.run(A, B, H, u, z)
        estimated_states.append(estimate)

    # Преобразование списков в массивы для удобства вычислений
    true_states = np.array(true_states).squeeze()
    estimated_states = np.array(estimated_states).squeeze()

    # Вычисление относительной погрешности
    relative_errors = np.linalg.norm(estimated_states - true_states, axis=1)/np.linalg.norm(estimated_states, axis=1)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(relative_errors, label="Relative Error", marker='o')
    plt.xlabel("Time step")
    plt.ylabel("Relative Error")
    plt.title("Relative Error Between Estimated and True States")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    true_states = np.array(true_states).squeeze()
    measurements = np.array(measurements).squeeze()
    estimated_states = np.array(estimated_states).squeeze()
    
    plt.figure(figsize=(10, 6))
    plt.plot(true_states[:, 0], label = "true state")
    plt.plot(measurements[:, 0], label = "measyrment state")
    plt.plot(estimated_states[:, 0], label = "estimated_states")
    plt.xlabel("Time step")
    plt.ylabel("value x")
    plt.title("Измерение, истенное значение, отфильтрованные данные")
    plt.legend()
    plt.tight_layout()
    plt.show()
