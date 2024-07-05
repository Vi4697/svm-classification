import numpy as np
from cvxopt import matrix, solvers

def quadratic_kernel(x1, x2):
    return (1 + np.dot(x1, x2)) ** 2

def prepare_qp_params(X, y, C):
    n_samples = X.shape[0]
    K = np.array([[quadratic_kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n_samples))
    G_std = matrix(np.diag(-np.ones(n_samples)))
    h_std = matrix(np.zeros(n_samples))
    G_slack = matrix(np.diag(np.ones(n_samples)))
    h_slack = matrix(np.ones(n_samples) * C)
    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.vstack((h_std, h_slack)))
    A = matrix(y.astype(np.double), (1, n_samples), 'd')
    b = matrix(0.0)
    return P, q, G, h, A, b

def fit_svm(X, y, C):
    P, q, G, h, A, b = prepare_qp_params(X, y, C)
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])
    return alphas

def calculate_bias(X, y, alphas):
    support_vectors = alphas > 1e-5
    ind = np.arange(len(alphas))[support_vectors]
    alphas_sv = alphas[support_vectors]
    sv_X = X[support_vectors]
    sv_y = y[support_vectors]
    b = np.mean(sv_y - np.sum(alphas_sv * sv_y * np.array([[quadratic_kernel(sv_X[i], sv_X[j]) for j in range(len(sv_X))] for i in range(len(sv_X))]), axis=1))
    return b

def predict_svm(X_train, y_train, alphas, X_test, b):
    support_vectors = alphas > 1e-5
    alphas_sv = alphas[support_vectors]
    sv_X = X_train[support_vectors]
    sv_y = y_train[support_vectors]
    y_predict = []
    for x in X_test:
        prediction = sum(alphas_sv * sv_y * np.array([quadratic_kernel(x, sv_X[i]) for i in range(len(sv_X))])) + b
        y_predict.append(np.sign(prediction))
    return np.array(y_predict)
