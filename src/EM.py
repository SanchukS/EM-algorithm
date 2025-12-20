import numpy as np
from typing import Optional, List, Self, Tuple
from sklearn.cluster import KMeans

class GMM:
    """
    Gaussian Mixture Model (GMM) с реализацией через EM-алгоритм.
    Для численной стабильности E-шаг выполняется в логарифмическом масштабе (Log-Sum-Exp).
    
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
    kmeans_start : bool, default=True
        Использовать ли K-Means для инициализации параметров.
    n_init : int, default=10
        Количество запусков K-Means при инициализации.
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
        
        self.gamma: Optional[np.ndarray] = None   # (N, K)
        self.log_likelihood_history: List[float] = [] 

    def _init_with_kmeans(self, X: np.ndarray) -> None:
        """Инициализация параметров через K-Means."""
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
        self.covs[:, np.arange(D), np.arange(D)] = 1.0
        self.weights = np.full(self.K, 1 / self.K)

    def _calc_log_pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисляет логарифм плотности вероятности (ln PDF) для каждого кластера.
        Используется вместо стандартного PDF для предотвращения underflow.
        
        Returns
        -------
        np.ndarray формы (N, K)
            Значения ln P(x | mu_k, sigma_k).
        """
        N, D = X.shape
        diff = X - self.means[:, None, :] # (K, N, D)
        
        # Решение системы для вычисления (x-mu)^T * Sigma^-1 * (x-mu)
        tmp = np.linalg.solve(self.covs, np.transpose(diff, axes=(0, 2, 1))) # (K, D, N)
        tmp = np.transpose(tmp, axes=(0, 2, 1)) # (K, N, D)
        
        mahalanobis_sq = np.sum(diff * tmp, axis=2).T # (N, K)
        
        # Вычисление логарифма определителя через slogdet для стабильности
        sign, logdet = np.linalg.slogdet(self.covs)
        
        const = -0.5 * (D * np.log(2 * np.pi) + logdet)
        return const - 0.5 * mahalanobis_sq
    
    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        E-шаг: Вычисление ответственности (Gamma) через Log-Sum-Exp.
        
        Returns
        -------
        gamma : np.ndarray
            Матрица вероятностей принадлежности кластерам (N, K).
        log_likelihood : float
            Суммарный лог-правдоподобие данных на текущем шаге.
        """
        # 1. Логарифм числителя: ln(pi_k) + ln(N(x|...))
        log_pdf = self._calc_log_pdf(X)
        log_weighted_pdf = log_pdf + np.log(self.weights + 1e-300) # (N, K)
        
        # 2. Log-Sum-Exp Trick для знаменателя: ln(sum(exp(x_i)))
        # a = max(x_i)
        max_log_weighted = np.max(log_weighted_pdf, axis=1, keepdims=True)
        
        # ln(sum(exp(x - a))) + a
        log_sum_exp = max_log_weighted + np.log(
            np.sum(np.exp(log_weighted_pdf - max_log_weighted), axis=1, keepdims=True)
        )
        
        # 3. Вычисление ln(gamma) = ln(числитель) - ln(знаменатель)
        log_gamma = log_weighted_pdf - log_sum_exp
        gamma = np.exp(log_gamma)
        
        return gamma, np.sum(log_sum_exp)

    def _update_weights(self, gamma: np.ndarray, N: int) -> np.ndarray:
        """Обновление весов кластеров."""
        N_k = np.sum(gamma, axis=0)
        return N_k / N
    
    def _update_means(self, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Обновление центров кластеров."""
        N_k = np.sum(gamma, axis=0)
        numerator = gamma.T @ X 
        return numerator / (N_k[:, None] + 1e-10)

    def _update_covs(self, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Обновление ковариационных матриц."""
        N, D = X.shape
        N_k = np.sum(gamma, axis=0)
        
        diff = X - self.means[:, None, :] # (K, N, D)
        
        # Векторизованное вычисление взвешенной ковариации
        numerator_part = gamma.T[:, :, None] * diff # (K, N, D)
        numerator_part = np.transpose(numerator_part, axes=(0, 2, 1)) # (K, D, N)
        numerator = numerator_part @ diff # (K, D, D)
        
        covs = numerator / (N_k[:, None, None] + 1e-10)

        # Регуляризация диагонали
        covs[:, np.arange(D), np.arange(D)] += self.r

        return covs

    def fit(self, X: np.ndarray) -> Self:
        """
        Обучение модели методом EM.
        """
        N, D = X.shape
        self.log_likelihood_history = []
        
        if self.kmeans_start:
            self._init_with_kmeans(X)
        else:
            self._init_random(X)
        
        for i in range(self.max_iter):
            # E-шаг (теперь возвращает и Log-Likelihood)
            self.gamma, log_likelihood = self._e_step(X)
            self.log_likelihood_history.append(log_likelihood)

            # Проверка сходимости
            if i > 0:
                diff = abs(log_likelihood - self.log_likelihood_history[-2])
                if diff < self.tol:
                    break

            # M-шаг
            self.weights = self._update_weights(self.gamma, N)
            self.means = self._update_means(X, self.gamma)
            self.covs = self._update_covs(X, self.gamma)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток кластеров."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей принадлежности к кластерам."""
        gamma, _ = self._e_step(X)
        return gamma