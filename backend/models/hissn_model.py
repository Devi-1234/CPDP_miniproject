import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler

# Function to run hissn model
def run_hissn(source_files, target_filename, test_filename, data_path, ml_model):
    # Load source datasets
    X_source_list, y_source_list = [], []
    for file in source_files:
        data = pd.read_csv(os.path.join(data_path, file))
        X_source_list.append(data.iloc[:, :-1])
        y_source_list.append(data.iloc[:, -1].astype(str).str.replace("b'", "").str.replace("'", "").astype(int).values)

    X_source = pd.concat(X_source_list, axis=0)
    y_source = np.hstack(y_source_list)

    # Load target and test datasets
    target_data = pd.read_csv(os.path.join(data_path, target_filename))
    test_data = pd.read_csv(os.path.join(data_path, test_filename))

    X_target = target_data.iloc[:, :-1]
    y_target = target_data.iloc[:, -1].astype(str).str.replace("b'", "").str.replace("'", "").astype(int).values
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1].astype(str).str.replace("b'", "").str.replace("'", "").astype(int).values

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

    # Apply Random Undersampling
    undersampler = RandomUnderSampler(random_state=42)
    X_source_bal, y_source_bal = undersampler.fit_resample(X_source_scaled, y_source)

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    }

    if ml_model not in models:
        raise ValueError(f"Unsupported ML model: {ml_model}")

    # Train and evaluate model
    model = models[ml_model]
    print(f"Training {ml_model}...")
    model.fit(X_source_bal, y_source_bal)

    # Get probabilities
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Compute metrics
    bacc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    gmean = np.sqrt(precision_score(y_test, y_pred) * recall_score(y_test, y_pred))

    # Print detailed metrics
    print("\nDetailed Metrics:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")

    results = {
        "Balanced Accuracy": bacc,
        "AUC": auc,
        "F1 Score": f1,
        "G-Mean": gmean
    }

    return results
