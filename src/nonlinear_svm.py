import numpy as np

def rbf_kernel(X1, X2, gamma):
    if X1.ndim == 1:
        X1 = X1[np.newaxis, :]
    if X2.ndim == 1:
        X2 = X2[np.newaxis, :]
    return np.exp(-gamma * np.sum((X1[:, np.newaxis] - X2[np.newaxis, :])**2, axis=2))

class KernelSVMModel:
    def __init__(self, learning_rate, lambda_param, n_iters, gamma, kernel=rbf_kernel):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.gamma = gamma
        self.kernel = kernel
        self.X = None
        self.alpha = None
        self.b = 0

    def fit(self, X, y):
        self.X = X
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        y_ = np.where(y <= 0, -1, 1)
        K = self.kernel(X, X, self.gamma)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y_[i] * (np.sum(self.alpha * y_ * K[:, i]) - self.b) >= 1
                if not condition:
                    self.alpha[i] += self.learning_rate * (1 - y_[i] * (np.sum(self.alpha * y_ * K[:, i]) - self.b))
                    self.b += self.learning_rate * y_[i]

    def predict(self, X):
        K = self.kernel(X, self.X, self.gamma)
        y_train_ = np.where(y_train <= 0, -1, 1)
        approx = np.dot(K, self.alpha * y_train_) - self.b
        return np.sign(approx)
