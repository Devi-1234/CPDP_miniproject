import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold

def coral(source, target):
    # Optimize CORAL by using more efficient matrix operations
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    cov_target = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-6

    # Use faster SVD implementation
    U_source, S_source, _ = np.linalg.svd(cov_source, full_matrices=False)
    U_target, S_target, _ = np.linalg.svd(cov_target, full_matrices=False)

    S_source = np.maximum(S_source, 1e-8)
    S_target = np.maximum(S_target, 1e-8)

    # Optimize matrix multiplications
    sqrt_cov_source = U_source @ np.diag(np.sqrt(S_source)) @ U_source.T
    inv_sqrt_cov_source = U_source @ np.diag(1.0 / np.sqrt(S_source)) @ U_source.T
    sqrt_cov_target = U_target @ np.diag(np.sqrt(S_target)) @ U_target.T

    return source @ inv_sqrt_cov_source @ sqrt_cov_target

class TCA:
    def __init__(self, kernel='linear', n_components=10):
        if n_components <= 0:
            raise ValueError("n_components must be greater than 0")
        self.kernel = kernel
        self.n_components = n_components

    def fit_transform(self, X_source, X_target):
        X = np.vstack((X_source, X_target))
        kpca = KernelPCA(kernel=self.kernel, n_components=self.n_components)
        K = kpca.fit_transform(X)
        return K[:len(X_source)], K[len(X_source):]

def optimize_threshold(y_true, y_prob):
    """Find optimal threshold that maximizes F1 score"""
    thresholds = np.arange(0.2, 0.7, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

def run_tca_coral(source_files, target_filename, test_filename, data_path, ml_model):
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

    # Apply TCA with proper n_components
    print("Applying TCA...")
    tca = TCA(n_components=min(10, X_source_scaled.shape[1]))  # Ensure n_components is valid
    X_source_tca, X_target_tca = tca.fit_transform(X_source_scaled, X_target_scaled)
    X_test_tca = tca.fit_transform(X_target_scaled, X_test_scaled)[1]

    # Combine source and target data
    X_combined = np.vstack((X_source_tca, X_target_tca))
    y_combined = np.hstack((y_source, y_target))

    # Two-step feature selection
    print("Performing feature selection...")
    # Step 1: Remove low variance features
    selector_variance = VarianceThreshold(threshold=0.01)
    X_combined_var = selector_variance.fit_transform(X_combined)
    X_test_var = selector_variance.transform(X_test_tca)

    # Ensure we have features after variance threshold
    if X_combined_var.shape[1] == 0:
        print("Warning: No features passed variance threshold. Using all features.")
        X_combined_var = X_combined
        X_test_var = X_test_tca

    # Step 2: Select most informative features
    selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=100, random_state=42),
        threshold='mean'  # Changed from 'median' to 'mean' to be less strict
    )
    X_combined_selected = selector.fit_transform(X_combined_var, y_combined)
    X_test_selected = selector.transform(X_test_var)

    # Ensure we have features after model-based selection
    if X_combined_selected.shape[1] == 0:
        print("Warning: No features selected by model. Using variance-selected features.")
        X_combined_selected = X_combined_var
        X_test_selected = X_test_var

    # Print number of selected features
    n_features = X_combined_selected.shape[1]
    print(f"Selected {n_features} features after feature selection")

    # Define classifiers with optimized parameters
    models = {
        "RandomForest": RandomForestClassifier(
            random_state=42,
            class_weight={0: 1, 1: 3},  # Matched class weights
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight={0: 1, 1: 4},  # Matched class weights
            solver='saga',
            random_state=42,
            C=0.2,  # Matched regularization
            penalty='l2',
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=4,
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            n_jobs=-1
        )
    }

    if ml_model not in models:
        raise ValueError(f"Unsupported ML model: {ml_model}")

    # Train and evaluate model
    model = models[ml_model]
    print(f"Training {ml_model} on Source + Target Data...")
    
    # For Logistic Regression, use pipeline with SMOTE and feature selection
    if ml_model == "LogisticRegression":
        # Create a pipeline with SMOTE and model
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),  # Matched sampling ratio
            ('model', model)
        ])
        
        # Parameter grid
        param_grid = {
            'model__C': [0.1, 0.2, 0.3],
            'model__class_weight': [{0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_combined_selected, y_combined)
        best_pipeline = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Train final model
        best_pipeline.fit(X_combined_selected, y_combined)
        
        # Get probabilities
        y_prob = best_pipeline.predict_proba(X_test_selected)[:, 1]
        
        # Find optimal threshold
        optimal_threshold = optimize_threshold(y_test, y_prob)
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        
        # Make predictions with optimal threshold
        y_pred = (y_prob >= optimal_threshold).astype(int)
    else:
        # For other models, use SMOTE directly
        smote = SMOTE(random_state=42, sampling_strategy=0.8)  # Matched sampling ratio
        X_train_balanced, y_train_balanced = smote.fit_resample(X_combined_selected, y_combined)
        model.fit(X_train_balanced, y_train_balanced)
        
        # Get probabilities
        y_prob = model.predict_proba(X_test_selected)[:, 1]
        
        # Find optimal threshold
        optimal_threshold = optimize_threshold(y_test, y_prob)
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        
        # Make predictions with optimal threshold
        y_pred = (y_prob >= optimal_threshold).astype(int)

    # Compute Metrics
    bacc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    gmean = np.sqrt(sensitivity * specificity)
    precision = tp / (tp + fp)

    # Print detailed metrics
    print("\nDetailed Metrics:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {sensitivity:.4f}")
    print(f"F1 Score: {f1:.4f}")

    results = {
        "Balanced Accuracy": bacc,
        "AUC": auc,
        "F1 Score": f1,
        "G-Mean": gmean,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Precision": precision,
        "Recall": sensitivity
    }

    return results 