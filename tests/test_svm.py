import unittest
from src.data_loader import load_and_preprocess_data
from src.linear_svm import SimpleSVM
from src.nonlinear_svm import KernelSVMModel, rbf_kernel
from src.quadratic_svm import fit_svm, calculate_bias, predict_svm, quadratic_kernel

class TestSVM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_path = 'data/wdbc.data'
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = load_and_preprocess_data(cls.file_path)

    def test_linear_svm(self):
        linear_svm = SimpleSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
        linear_svm.fit(self.X_train, self.y_train)
        y_pred_linear = linear_svm.predict(self.X_test)
        y_pred_linear = np.where(y_pred_linear == -1, 0, y_pred_linear)
        accuracy_linear = np.mean(self.y_test == y_pred_linear)
        self.assertGreaterEqual(accuracy_linear, 0.9)

    def test_nonlinear_svm_rbf(self):
        nonlinear_svm = KernelSVMModel(learning_rate=0.001, lambda_param=0.01, n_iters=20000, gamma=0.01)
        nonlinear_svm.fit(self.X_train, self.y_train)
        y_pred_nonlinear = nonlinear_svm.predict(self.X_test)
        y_pred_nonlinear = np.where(y_pred_nonlinear == -1, 0, y_pred_nonlinear)
        accuracy_nonlinear = np.mean(self.y_test == y_pred_nonlinear)
        self.assertGreaterEqual(accuracy_nonlinear, 0.9)

    def test_quadratic_svm(self):
        y_train_ = np.where(self.y_train <= 0, -1, 1)
        C = 1.0
        alphas = fit_svm(self.X_train, y_train_, C)
        b = calculate_bias(self.X_train, y_train_, alphas)
        y_pred = predict_svm(self.X_train, y_train_, alphas, self.X_test, b)
        y_pred = np.where(y_pred == -1, 0, y_pred)
        accuracy = np.mean(self.y_test == y_pred)
        self.assertGreaterEqual(accuracy, 0.9)

if __name__ == '__main__':
    unittest.main()
