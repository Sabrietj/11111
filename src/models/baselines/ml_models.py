import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class AppScannerBaseline:
    def __init__(self, random_state=42):
        self.model_name = "AppScanner"
        self.expected_dim = 54
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                class_weight='balanced',
                n_jobs=-1,
                random_state=random_state
            ))
        ])

    def fit(self, X, y):
        if X.shape[1] != self.expected_dim:
            print(f"[警告] AppScanner 预期 54 维特征，但输入为 {X.shape[1]} 维。")
        X = np.nan_to_num(X)
        self.pipeline.fit(X, y)

    def predict(self, X):
        X = np.nan_to_num(X)
        return self.pipeline.predict(X)

class FlowmeterRFBaseline:
    def __init__(self, random_state=42):
        self.model_name = "FlowmeterRF"
        self.expected_dim = 81
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                class_weight='balanced',
                n_jobs=-1,
                random_state=random_state
            ))
        ])

    def fit(self, X, y):
        if X.shape[1] != self.expected_dim:
            print(f"[警告] FlowmeterRF 预期 81 维特征，但输入为 {X.shape[1]} 维。")
        X = np.nan_to_num(X)
        self.pipeline.fit(X, y)

    def predict(self, X):
        X = np.nan_to_num(X)
        return self.pipeline.predict(X)