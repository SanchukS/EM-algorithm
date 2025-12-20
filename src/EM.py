import numpy as np
from typing import Optional, List, Self
from sklearn.cluster import KMeans

class GMM:
    """
    Gaussian Mixture Model (GMM) с реализацией через EM-алгоритм.
    
    Parameters
    ----------
    n_components : int
        Количество кластеров (K).
    max_iter : int, default=100
        Максимальное количество итераций алгоритма.
    tol : float, default=1e-4
        Порог сходимости по изменению Log-Likelihood.
    r : float, default=1e-6
        Коэффициент регуляризации для добавления на диагональ ковариационной матрицы.
        Предотвращает сингулярность (вырожденность) матриц.
    kmeans_start : bool, default=True
        Использовать ли K-Means для инициализации параметров.
        Если False, используется случайная инициализация.
    n_init : int, default=10
        Количество запусков K-Means (если используется).
    """

    def __init__(
            self, 
            n_components: int, 
            max_iter: int = 100, 
            tol: float = 1e-4, 
            r: float = 1e-6, 
            kmeans_start: bool = True, 
            n_init: int = 10
        ) -> None:
        
        self.K: int = n_components
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.r: float = r
        self.kmeans_start: bool = kmeans_start
        self.n_init: int = n_init
        
        # Обучаемые параметры
        self.means: Optional[np.ndarray] = None   # (K, D)
        self.covs: Optional[np.ndarray] = None    # (K, D, D)
        self.weights: Optional[np.ndarray] = None # (K,)
        
        # История сходимости
        self.log_likelihood_history: List[float] = [] 

    def _init_with_kmeans(self, X: np.ndarray) -> None:
        """Инициализация параметров через K-Means для лучшей сходимости."""
        N, D = X.shape
        kmeans = KMeans(n_clusters=self.K, n_init=self.n_init, random_state=42)
        labels = kmeans.fit_predict(X)
        
        self.means = kmeans.cluster_centers_
        self.weights = np.zeros(self.K)
        self.covs = np.zeros((self.K, D, D))
        
        for k in range(self.K):
            X_k = X[labels == k]
            self.weights[k] = len(X_k) / N
            
            if len(X_k) > 1:
                self.covs[k] = np.cov(X_k, rowvar=False)
            else:
                self.covs[k] = np.eye(D)
            
            self.covs[k] += np.eye(D) * self.r

    def _init_random(self, X: np.ndarray) -> None:
        """Случайная инициализация параметров."""
        N, D = X.shape
        mean_indexes = np.random.choice(N, self.K, replace=False)
        self.means = X[mean_indexes]
        self.covs = np.zeros(shape=(self.K, D, D))
        # Инициализация единичными матрицами
        self.covs[:, np.arange(D), np.arange(D)] = 1
        self.weights = np.full(self.K, 1 / self.K)

    def _calc_pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисляет плотность вероятности (PDF) для каждого кластера.
        
        Returns
        -------
        np.ndarray формы (N, K)
            Значения P(x | mu_k, sigma_k) для каждой точки и кластера.
        """
        N, D = X.shape
        diff = X - self.means[:, None, :] # Broadcasting: (K, N, D)
        
        # Вычисление (x-mu)^T * Sigma^-1 * (x-mu)
        # solve решает систему Ax = B, что эквивалентно inv(A) @ B
        tmp = np.linalg.solve(self.covs, np.transpose(diff, axes=(0, 2, 1))) # (K, D, N)
        tmp = np.transpose(tmp, axes=(0, 2, 1)) # (K, N, D)
        
        mahalanobis_sq = np.sum(diff * tmp, axis=2).T # (N, K)
        
        # Нормировочный коэффициент
        det = np.linalg.det(self.covs)
        # Защита от отрицательного определителя (численная погрешность)
        det = np.maximum(det, 1e-10) 
        fraction = 1 / np.sqrt((2 * np.pi) ** D * det) # (K,)
        
        return fraction * np.exp(-mahalanobis_sq / 2)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-шаг: Вычисление матрицы ответственности (Gamma).
        Gamma_nk = P(z_n = k | x_n)
        """
        pdfs = self._calc_pdf(X)
        numerator = pdfs * self.weights # P(x|k) * P(k)
        denominator = np.sum(numerator, axis=1, keepdims=True) # P(x)
        
        # Добавляем epsilon для стабильности деления
        return numerator / (denominator + 1e-8)

    def _update_weights(self, gamma: np.ndarray, N: int) -> np.ndarray:
        """Обновление весов кластеров (pi_k)."""
        N_k = np.sum(gamma, axis=0)
        return N_k / N
    
    def _update_means(self, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Обновление центров кластеров (mu_k)."""
        N_k = np.sum(gamma, axis=0)
        # Взвешенная сумма координат
        numerator = gamma.T @ X 
        return numerator / (N_k[:, None] + 1e-8)

    def _update_covs(self, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Обновление ковариационных матриц (Sigma_k)."""
        N, D = X.shape
        N_k = np.sum(gamma, axis=0)
        
        diff = X - self.means[:, None, :] # (K, N, D)
        
        # Векторизованное вычисление взвешенной ковариации
        # Эквивалент: sum(gamma * (x-mu)(x-mu)^T)
        numerator_part = gamma.T[:, :, None] * diff # (K, N, D)
        numerator_part = np.transpose(numerator_part, axes=(0, 2, 1)) # (K, D, N)
        numerator = numerator_part @ diff # (K, D, D)
        
        covs = numerator / (N_k[:, None, None] + 1e-8)

        # Регуляризация диагонали
        covs[:, np.arange(D), np.arange(D)] += self.r

        return covs

    def fit(self, X: np.ndarray) -> Self:
        """
        Запуск обучения модели.
        
        Parameters
        ----------
        X : np.ndarray формы (N, D)
            Входные данные.
        """
        N, D = X.shape
        self.log_likelihood_history = []
        
        # 1. Инициализация
        if self.kmeans_start:
            self._init_with_kmeans(X)
        else:
            self._init_random(X)
        
        for i in range(self.max_iter):
            # 2. E-шаг
            self.gamma = self._e_step(X)

            # 3. M-шаг
            self.weights = self._update_weights(self.gamma, N)
            self.means = self._update_means(X, self.gamma)
            self.covs = self._update_covs(X, self.gamma)

            # 4. Оценка сходимости (Log-Likelihood)
            pdfs = self._calc_pdf(X)
            # P(X) = sum(w_k * P(X|k))
            total_prob = np.sum(pdfs * self.weights, axis=1)
            MLL = np.sum(np.log(total_prob + 1e-8))
            
            self.log_likelihood_history.append(MLL)

            if len(self.log_likelihood_history) > 1:
                diff = abs(MLL - self.log_likelihood_history[-2])
                if diff < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание жестких меток кластеров."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей принадлежности к кластерам."""
        return self._e_step(X)