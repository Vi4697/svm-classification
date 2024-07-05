# SVM Classification

This project demonstrates the implementation of different Support Vector Machine (SVM) classifiers for binary classification using the Breast Cancer Wisconsin Diagnostic dataset.

## Dataset

The dataset used for this project is the Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository.

- **Source**: [Breast Cancer Wisconsin Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Project Structure

```plaintext
svm-classification/
│
├── data/
│   └── wdbc.data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── linear_svm.py
│   ├── nonlinear_svm.py
│   ├── quadratic_svm.py
│   └── main.py
│
├── tests/
│   ├── __init__.py
│   └── test_svm.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup and Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/svm-classification.git
    cd svm-classification
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Ensure dataset is in the `data/` directory**.

## Usage

To run the SVM classification models, execute the `main.py` script:

```sh
python src/main.py
```

## Code Overview

### 1. Data Loading and Preprocessing

The data loading and preprocessing functions are in `data_loader.py`.

### 2. Linear SVM

The implementation of the linear SVM is in `linear_svm.py`.

### 3. Nonlinear SVM with RBF Kernel

The implementation of the nonlinear SVM with RBF kernel is in `nonlinear_svm.py`.

### 4. Nonlinear SVM with Quadratic Kernel

The implementation of the nonlinear SVM with quadratic kernel is in `quadratic_svm.py`.

### 5. Main Function

The main orchestration of the code is in `main.py`.

## Testing

To run tests, use:
```sh
python -m unittest discover tests
```

## Author

- Viktoriia Zvonarova

## License

This project is licensed under the MIT License.
