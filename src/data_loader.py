import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path):
    column_names = [
        'ID', 'Diagnosis', 'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean',
        'Smoothness_mean', 'Compactness_mean', 'Concavity_mean', 'Concave_points_mean',
        'Symmetry_mean', 'Fractal_dimension_mean', 'Radius_se', 'Texture_se', 'Perimeter_se',
        'Area_se', 'Smoothness_se', 'Compactness_se', 'Concavity_se', 'Concave_points_se',
        'Symmetry_se', 'Fractal_dimension_se', 'Radius_worst', 'Texture_worst', 'Perimeter_worst',
        'Area_worst', 'Smoothness_worst', 'Compactness_worst', 'Concavity_worst',
        'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst'
    ]

    data = pd.read_csv(file_path, header=None, names=column_names)
    data = data.drop(columns=['ID'])
    label_encoder = LabelEncoder()
    data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])  # M=1, B=0

    train_size = 25
    test_size = 25

    belonging_diagnosis_train = data[data["Diagnosis"] == 1].head(train_size)
    not_belonging_diagnosis_train = data[data["Diagnosis"] == 0].head(train_size)
    train_data = pd.concat([belonging_diagnosis_train, not_belonging_diagnosis_train])

    belonging_diagnosis_test = data[data["Diagnosis"] == 1].tail(test_size)
    not_belonging_diagnosis_test = data[data["Diagnosis"] == 0].tail(test_size)
    test_data = pd.concat([belonging_diagnosis_test, not_belonging_diagnosis_test])

    X_train = train_data.drop(columns=['Diagnosis'])
    y_train = train_data['Diagnosis']
    X_test = test_data.drop(columns=['Diagnosis'])
    y_test = test_data['Diagnosis']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
