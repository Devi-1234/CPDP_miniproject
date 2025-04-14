import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

class TCA:
    def __init__(self, kernel='linear', n_components=10):
        self.kernel = kernel
        self.n_components = n_components

    def fit_transform(self, X_source, X_target):
        X = np.vstack((X_source, X_target))
        kpca = KernelPCA(kernel=self.kernel, n_components=self.n_components)
        K = kpca.fit_transform(X)
        return K[:len(X_source)], K[len(X_source):]

def run_tca(source_files, target_filename, test_filename, data_path, ml_model):
    # Load source datasets
    X_source_list, y_source_list = [], []
    for file in source_files:
        data = pd.read_csv(os.path.join(data_path, file))
        X_source_list.append(data.iloc[:, :-1])
        y_source_list.append(data.iloc[:, -1].astype(str).str.strip("b'").astype(int).values)

    X_source = pd.concat(X_source_list, axis=0)
    y_source = np.hstack(y_source_list)

    # Load target and test datasets
    target_data = pd.read_csv(os.path.join(data_path, target_filename))
    test_data = pd.read_csv(os.path.join(data_path, test_filename))

    X_target = target_data.iloc[:, :-1]
    y_target = target_data.iloc[:, -1].astype(str).str.strip("b'").astype(int).values
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1].astype(str).str.strip("b'").astype(int).values

    # Get common features
    common_features = X_source.columns.intersection(X_target.columns)
    X_source = X_source[common_features]
    X_target = X_target[common_features]
    X_test = X_test[common_features]

    # Convert to NumPy
    X_source, X_target, X_test = X_source.values, X_target.values, X_test.values

    # Scale data
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    X_test_scaled = scaler.transform(X_test)

    # Apply TCA
    print("Applying TCA...")
    tca = TCA()
    X_source_tca, X_target_tca = tca.fit_transform(X_source_scaled, X_target_scaled)
    X_test_tca = tca.fit_transform(X_target_scaled, X_test_scaled)[1]

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_source_tca_resampled, y_source_resampled = smote.fit_resample(X_source_tca, y_source)

    # Define models with parameters matching the actual implementation
    models = {
        "RandomForest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            random_state=42,
            max_iter=1000,  # Reduced from 3000 to match actual implementation
            class_weight='balanced',
            solver='lbfgs',  # Changed from 'saga' to 'lbfgs'
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            n_jobs=-1,
            tree_method='hist'
        )
    }

    if ml_model not in models:
        raise ValueError(f"Unsupported ML model: {ml_model}")

    # Hyperparameter tuning with GridSearchCV
    param_grid_xgb = {
        'n_estimators': [50, 100],
        'max_depth': [3, 4],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'scale_pos_weight': [1, 2, 3]  # Added scale_pos_weight to grid search
    }

    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
    }

    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }

    model = models[ml_model]

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if ml_model == "XGBoost":
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_xgb, cv=cv, n_jobs=-1, scoring='balanced_accuracy', verbose=2)
    elif ml_model == "RandomForest":
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_rf, cv=cv, n_jobs=-1, scoring='balanced_accuracy', verbose=2)
    elif ml_model == "LogisticRegression":
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_lr, cv=cv, n_jobs=-1, scoring='balanced_accuracy', verbose=2)

    print(f"Training and tuning {ml_model}...")
    grid_search.fit(X_source_tca_resampled, y_source_resampled)
    best_model = grid_search.best_estimator_

    # Get the best model's predictions
    y_prob = best_model.predict_proba(X_test_tca)[:, 1]
    y_pred = best_model.predict(X_test_tca)

    # Compute metrics
    auc = roc_auc_score(y_test, y_prob)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    gmean = np.sqrt(f1 * bacc)

    # Print detailed metrics
    print("\nDetailed Metrics:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"G-Mean: {gmean:.4f}")

    results = {
        "Balanced Accuracy": bacc,
        "AUC": auc,
        "F1 Score": f1,
        "G-Mean": gmean
    }

    return results
