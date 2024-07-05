import numpy as np
from src.data_loader import load_and_preprocess_data
from src.linear_svm import SimpleSVM
from src.nonlinear_svm import KernelSVMModel, rbf_kernel
from src.quadratic_svm import fit_svm, calculate_bias, predict_svm, quadratic_kernel

def main():
    file_path = 'data/wdbc.data'
    X_train, y_train, X_test, y_test = load_and_preprocess_data(file_path)

    # Linear SVM
    linear_svm = SimpleSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    linear_svm.fit(X_train, y_train)
    y_pred_linear = linear_svm.predict(X_test)
    y_pred_linear = np.where(y_pred_linear == -1, 0, y_pred_linear)
    accuracy_linear = np.mean(y_test == y_pred_linear)
    print(f"Accuracy of Linear SVM: {accuracy_linear}")

    # Nonlinear SVM with RBF Kernel
    learning_rates = [0.001, 0.01]
    lambda_params = [0.01, 0.1]
    gammas = [0.001, 0.01, 0.1]
    n_iters = 20000

    best_params, best_accuracy = manual_grid_search(X_train, y_train, X_test, y_test, learning_rates, lambda_params, gammas, n_iters)
    nonlinear_svm = KernelSVMModel(learning_rate=best_params['learning_rate'], lambda_param=best_params['lambda_param'], n_iters=n_iters, gamma=best_params['gamma'])
    nonlinear_svm.fit(X_train, y_train)
    y_pred_nonlinear = nonlinear_svm.predict(X_test)
    y_pred_nonlinear = np.where(y_pred_nonlinear == -1, 0, y_pred_nonlinear)
    accuracy_nonlinear = np.mean(y_test == y_pred_nonlinear)
    print(f"Accuracy of Nonlinear SVM (RBF kernel) with best params: {accuracy_nonlinear}")

    # Nonlinear SVM with Quadratic Kernel
    y_train_ = np.where(y_train <= 0, -1, 1)
    C = 1.0
    alphas = fit_svm(X_train, y_train_, C)
    b = calculate_bias(X_train, y_train_, alphas)
    y_pred = predict_svm(X_train, y_train_, alphas, X_test, b)
    y_pred = np.where(y_pred == -1, 0, y_pred)
    accuracy = np.mean(y_test == y_pred)
    print(f"Accuracy of Nonlinear SVM with Quadratic Kernel: {accuracy}")

    # Compare results
    print(f"Linear SVM Accuracy: {accuracy_linear}")
    print(f"Nonlinear SVM (RBF kernel) Accuracy: {accuracy_nonlinear}")
    print(f"Nonlinear SVM (Quadratic kernel, QP) Accuracy: {accuracy}")

    if accuracy_linear > accuracy_nonlinear and accuracy_linear > accuracy:
        print("Linear SVM performed better than Nonlinear SVMs (RBF and Quadratic kernel).")
    elif accuracy_nonlinear > accuracy_linear and accuracy_nonlinear > accuracy:
        print("Nonlinear SVM (RBF kernel) performed better than Linear SVM and Nonlinear SVM (Quadratic kernel).")
    else:
        print("Nonlinear SVM (Quadratic kernel) performed better than Linear SVM and Nonlinear SVM (RBF kernel).")

def manual_grid_search(X_train, y_train, X_test, y_test, learning_rates, lambda_params, gammas, n_iters):
    best_accuracy = 0
    best_params = {}

    for lr in learning_rates:
        for lp in lambda_params:
            for gm in gammas:
                model = KernelSVMModel(learning_rate=lr, lambda_param=lp, n_iters=n_iters, gamma=gm)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred = np.where(y_pred == -1, 0, y_pred)
                accuracy = np.mean(y_test == y_pred)
                print(f"LR: {lr}, Lambda: {lp}, Gamma: {gm} - Accuracy: {accuracy}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'learning_rate': lr, 'lambda_param': lp, 'gamma': gm}

    return best_params, best_accuracy

if __name__ == "__main__":
    main()
