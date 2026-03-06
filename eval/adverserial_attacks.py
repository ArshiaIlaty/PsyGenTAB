import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MIAEvaluator:
    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sample_size: int = None,
    ):
        """
        Initialize the Evaluator.
        
        NOTE: This evaluates 'Distinguishability'. 
        - AUC ~ 0.5: Synthetic data is indistinguishable from real (High Privacy/Fidelity).
        - AUC ~ 1.0: Synthetic data is easily distinguishable (Low Fidelity).
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.encoders = {}
        
        # Sample equal numbers from both datasets
        if sample_size is None:
            sample_size = min(len(real_data), len(synthetic_data))

        # Sample from real data
        if len(real_data) > sample_size:
            self.real_data = real_data.sample(n=sample_size, random_state=42)

        # Sample from synthetic data
        if len(synthetic_data) > sample_size:
            self.synthetic_data = synthetic_data.sample(n=sample_size, random_state=42)

        logger.info(f"Sample size used per dataset: {sample_size}")

    def _encode_categorical(self, df):
        """Encode categorical columns using LabelEncoder (Best for Random Forest)."""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype(str)
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                # Fit on all unique values from both datasets
                all_values = pd.concat(
                    [self.real_data[col].astype(str), self.synthetic_data[col].astype(str)]
                ).unique()
                self.encoders[col].fit(all_values)

            # Transform the column
            # Handle unseen labels by mapping them to a default if necessary (simple approach used here)
            # For robustness in production, you might want to use a dedicated encoder library
            df_encoded[col] = df_encoded[col].apply(
                lambda x: self.encoders[col].transform([x])[0] if x in self.encoders[col].classes_ else -1
            )

        return df_encoded

    def _to_one_hot(self, df):
        """Convert Label Encoded Data to One-Hot (Best for Linear/NN Models)."""
        # We assume df is already Label Encoded via _encode_categorical
        # We identify columns that were originally categorical
        categorical_cols = self.encoders.keys()
        
        # Filter to only cols present in df
        cols_to_encode = [c for c in categorical_cols if c in df.columns]
        
        if not cols_to_encode:
            return df
            
        # Use pandas get_dummies for simplicity and robustness
        # Note: In a strict pipeline, one should fit a OneHotEncoder, but for evaluation
        # pandas dummies is often sufficient provided the domain is consistent.
        df_ohe = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        return df_ohe.astype(float)

    def _impute_missing(self, df):
        """Impute missing values: mean for numeric, mode for categorical."""
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    # For object/categorical, fill with mode
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def prepare_data(self, test_size=0.2):
        """Prepare data with balanced classes."""
        # Encode categorical variables (Label Encoding)
        real_encoded = self._encode_categorical(self.real_data)
        synth_encoded = self._encode_categorical(self.synthetic_data)

        # Add membership labels (1=Real, 0=Synthetic)
        real_encoded["membership"] = 1
        synth_encoded["membership"] = 0

        # Combine datasets
        combined_data = pd.concat([real_encoded, synth_encoded], ignore_index=True)

        # Shuffle
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and test
        X = combined_data.drop("membership", axis=1)
        y = combined_data["membership"]
        
        # Stratified split ensures equal ratio of real/synthetic in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Recombine for the internal methods that expect full dataframes
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        return train_data, test_data

    def train_black_box_attack(self, train_data, test_data):
        """Train a Random Forest (Tree-based, handles Label Encoding well)."""
        X_train = self._impute_missing(train_data.drop("membership", axis=1))
        y_train = train_data["membership"]
        X_test = self._impute_missing(test_data.drop("membership", axis=1))
        y_test = test_data["membership"]

        # Base Classifier
        base_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        # Calibrated Classifier
        calibrated_clf = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
        calibrated_clf.fit(X_train, y_train)

        # Predictions
        y_pred = calibrated_clf.predict(X_test)
        y_proba = calibrated_clf.predict_proba(X_test)[:, 1]

        # Metrics
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5
        accuracy = accuracy_score(y_test, y_pred)

        # Feature Importance (Extract from the base estimator inside CalibratedClassifierCV)
        # Note: CalibratedClassifierCV.calibrated_classifiers_ holds the fitted base estimators
        # We fit a fresh base_clf just to get feature importances for reporting
        base_clf_for_imp = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        base_clf_for_imp.fit(X_train, y_train)
        
        return auc, accuracy, base_clf_for_imp.feature_importances_, X_train.columns, calibrated_clf

    def train_logistic_regression_attack(self, train_data, test_data):
        """Train a Logistic Regression (Linear, requires One-Hot Encoding)."""
        # Convert to One-Hot
        train_ohe = self._to_one_hot(train_data.drop("membership", axis=1))
        test_ohe = self._to_one_hot(test_data.drop("membership", axis=1))
        
        # Align columns (ensure train and test have same dummy columns)
        train_ohe, test_ohe = train_ohe.align(test_ohe, join='inner', axis=1)
        
        X_train = self._impute_missing(train_ohe)
        y_train = train_data["membership"]
        X_test = self._impute_missing(test_ohe)
        y_test = test_data["membership"]

        lr_clf = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear')
        lr_clf.fit(X_train, y_train)

        y_pred = lr_clf.predict(X_test)
        y_proba = lr_clf.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5
        accuracy = accuracy_score(y_test, y_pred)

        return auc, accuracy, lr_clf.coef_[0], X_train.columns, lr_clf

    def train_white_box_attack(self, train_data, test_data):
        """Train a Neural Network (Requires One-Hot Encoding + Standardization)."""
        
        # Convert to One-Hot
        train_ohe = self._to_one_hot(train_data.drop("membership", axis=1))
        test_ohe = self._to_one_hot(test_data.drop("membership", axis=1))
        
        # Align columns
        train_ohe, test_ohe = train_ohe.align(test_ohe, join='inner', axis=1)
        
        X_train_np = self._impute_missing(train_ohe).values.astype(np.float32)
        y_train_np = train_data["membership"].values.astype(np.float32)
        X_test_np = self._impute_missing(test_ohe).values.astype(np.float32)
        y_test_np = test_data["membership"].values.astype(np.float32)

        # Select Device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Create Tensors
        X_train_t = torch.FloatTensor(X_train_np).to(device)
        y_train_t = torch.FloatTensor(y_train_np).to(device)
        X_test_t = torch.FloatTensor(X_test_np).to(device)
        y_test_t = torch.FloatTensor(y_test_np).to(device)

        class EnhancedMIA_Net(nn.Module):
            def __init__(self, input_dim):
                super(EnhancedMIA_Net, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.network(x)

        model = EnhancedMIA_Net(X_train_t.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        model.train()
        for epoch in range(50): # Reduced epochs for speed
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_proba = model(X_test_t).cpu().numpy().flatten()
            y_pred = (y_proba > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_test_np, y_proba)
        except ValueError:
            auc = 0.5
        accuracy = accuracy_score(y_test_np, y_pred)

        return auc, accuracy, model

    def stratified_analysis(self, train_data, test_data):
        """Perform stratified analysis by different demographic groups."""
        stratified_results = {}
        strat_cols = ["gender", "hypertension", "heart_disease", "diabetes", "race", "sex"]

        # Only use cols that actually exist in the data
        valid_cols = [c for c in strat_cols if c in train_data.columns]

        for col in valid_cols:
            stratified_results[col] = {}
            unique_vals = train_data[col].unique()

            for val in unique_vals:
                test_mask = test_data[col] == val
                test_stratum = test_data[test_mask]

                if len(test_stratum) < 10:  # Skip very small strata
                    continue
                
                # Check if we have both classes in this stratum
                if len(test_stratum["membership"].unique()) < 2:
                    continue

                X_test_stratum = test_stratum.drop("membership", axis=1)
                X_test_imputed = self._impute_missing(X_test_stratum)
                y_test_stratum = test_stratum["membership"]

                # We train a quick lightweight RF just for this stratum to see separability
                # Ideally, we should use the GLOBAL model to test the stratum, 
                # but training a local model checks "local distinguishability"
                try:
                    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    # We need a small train set for this stratum
                    train_mask = train_data[col] == val
                    train_stratum = train_data[train_mask]
                    if len(train_stratum) < 10: continue
                    
                    X_train_strat = self._impute_missing(train_stratum.drop("membership", axis=1))
                    y_train_strat = train_stratum["membership"]
                    
                    clf.fit(X_train_strat, y_train_strat)
                    y_proba = clf.predict_proba(X_test_imputed)[:, 1]
                    auc = roc_auc_score(y_test_stratum, y_proba)

                    stratified_results[col][str(val)] = {
                        "auc": float(auc),
                        "sample_size": int(len(test_stratum))
                    }
                except Exception as e:
                    stratified_results[col][str(val)] = {"error": str(e)}

        return stratified_results

    def bootstrap_auc(self, model, X_test, y_true, n_bootstrap=1000):
        """
        Calculate bootstrap confidence intervals using a PRE-TRAINED model.
        Fixes the data leakage issue where the model was retrained on test data.
        """
        y_proba = model.predict_proba(X_test)[:, 1]
        n_samples = len(y_true)
        bootstrap_aucs = []
        
        # Convert to numpy for speed
        y_true_np = np.array(y_true)

        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true_np[indices]
            y_proba_boot = y_proba[indices]

            # Only calculate AUC if both classes are present
            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                auc_boot = roc_auc_score(y_true_boot, y_proba_boot)
                bootstrap_aucs.append(auc_boot)
            except ValueError:
                pass

        if not bootstrap_aucs:
            return {"mean_auc": 0.5, "ci_lower": 0.5, "ci_upper": 0.5}

        return {
            "mean_auc": float(np.mean(bootstrap_aucs)),
            "ci_lower": float(np.percentile(bootstrap_aucs, 2.5)),
            "ci_upper": float(np.percentile(bootstrap_aucs, 97.5)),
            "std_auc": float(np.std(bootstrap_aucs)),
        }

    def evaluate(self):
        """Evaluate MIA vulnerability / Distinguishability."""
        train_data, test_data = self.prepare_data()

        print("\n[Diagnostics] Data Shapes:")
        print(f"  Train: {train_data.shape}")
        print(f"  Test:  {test_data.shape}")

        # 1. Black Box Attack (Random Forest)
        # We capture the `trained_rf_model` here to use it for bootstrapping later
        (
            bb_auc,
            bb_acc,
            rf_importances,
            rf_feat_names,
            trained_rf_model 
        ) = self.train_black_box_attack(train_data, test_data)

        # 2. Logistic Regression Attack
        (
            lr_auc,
            lr_acc,
            lr_coefs,
            lr_feat_names,
            _
        ) = self.train_logistic_regression_attack(train_data, test_data)

        # 3. White Box Attack (Neural Net)
        try:
            wb_auc, wb_acc, _ = self.train_white_box_attack(train_data, test_data)
        except Exception as e:
            logger.error(f"White-box attack failed: {e}")
            wb_auc, wb_acc = bb_auc, bb_acc

        # 4. Stratified Analysis
        strat_results = self.stratified_analysis(train_data, test_data)

        # 5. Bootstrap Confidence Intervals (Fixing the leakage)
        # We use the X_test that matches the RF model (Label Encoded, Imputed)
        X_test_rf = self._impute_missing(test_data.drop("membership", axis=1))
        y_test = test_data["membership"]
        
        bootstrap_results = self.bootstrap_auc(trained_rf_model, X_test_rf, y_test)

        # 6. Final Score
        # Average of the three methods
        vulnerability_score = (bb_auc + lr_auc + wb_auc) / 3

        # Formatting results (convert numpy types for JSON serialization)
        feature_importance = dict(zip(rf_feat_names, rf_importances))
        results = {
            "summary_metrics": {
                "black_box_auc": float(bb_auc),
                "logistic_regression_auc": float(lr_auc),
                "white_box_auc": float(wb_auc),
                "overall_distinguishability_score": float(vulnerability_score),
            },
            "bootstrap_ci": bootstrap_results,
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "stratified_analysis": _json_safe(strat_results),
        }
        return results


def _json_safe(obj):
    """Convert nested dicts/lists with numpy types to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


# Target (outcome) column per dataset. Excluded from MIA so AUC reflects feature-based
# distinguishability, not trivial separation from synthetic target imbalance (e.g. vn_banking).
DATASET_TARGET_COLUMN = {
    "adult_census_income": "Target",
    "diabetes_health_indicators": "Diabetes_binary",
    "breast_cancer": "Class",
    "parkinsons": "status",
    "obesity": "NObeyesdad",
    "vn_banking": "is_churned",
    "lung_cancer": "class",
    "hypothyroid": "binaryClass",
    "liver_disorders": "selector",
    "heart_failure_clinical_records": "DEATH_EVENT",
    "pir_vision_office": "Target",
}


def align_real_synthetic_columns(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align column names and sets so real and synthetic are comparable (avoids trivial
    separation from ID columns or target-name mismatches like Target vs is_churned/Class).
    """
    real = real_df.copy()
    synth = synth_df.copy()
    # Rename synthetic target to match real when it's a known mismatch
    if "Target" in synth.columns and "Target" not in real.columns:
        if "is_churned" in real.columns:
            synth = synth.rename(columns={"Target": "is_churned"})
        elif "Class" in real.columns:
            synth = synth.rename(columns={"Target": "Class"})
    # Drop identifier-only columns from real so we compare on common features + target
    id_like = {"customer_id", "ID", "id"}
    drop_from_real = [c for c in real.columns if c in id_like and c not in synth.columns]
    if drop_from_real:
        real = real.drop(columns=drop_from_real)
    # Use only common columns (same set in both)
    common = list(real.columns.intersection(synth.columns))
    if not common:
        raise ValueError("No common columns after alignment")
    return real[common].copy(), synth[common].copy()


def normalize_boolean_like_columns(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize boolean-like columns so real and synthetic use the same encoding (e.g. t/f and 0/1 -> 0/1).
    Reduces trivial separation from dtype/encoding differences (e.g. hypothyroid).
    """
    real = real_df.copy()
    synth = synth_df.copy()
    bool_like = {"t", "f", "0", "1", "true", "false", "?", "nan", ""}

    def to_binary(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.lower()
        out = np.full(len(series), -1, dtype=np.float64)  # -1 = missing
        out = np.where(s.isin(("t", "true", "1")), 1, out)
        out = np.where(s.isin(("f", "false", "0")), 0, out)
        # Catch numeric 0/1 that became "0.0"/"1.0" or int 0/1
        numeric = pd.to_numeric(series, errors="coerce")
        out = np.where((out == -1) & (numeric == 1), 1, out)
        out = np.where((out == -1) & (numeric == 0), 0, out)
        return pd.Series(out, index=series.index).astype(int)

    for col in real.columns:
        if col not in synth.columns:
            continue
        r_vals = set(real[col].dropna().astype(str).str.strip().str.lower().unique())
        s_vals = set(synth[col].dropna().astype(str).str.strip().str.lower().unique())
        all_vals = r_vals | s_vals
        if not all_vals.issubset(bool_like):
            continue
        real[col] = to_binary(real[col])
        synth[col] = to_binary(synth[col])
    return real, synth


def coerce_numeric_columns(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coerce columns to numeric when one side is numeric and the other is object (e.g. age as string vs float).
    Reduces trivial separation from dtype differences (e.g. hypothyroid age).
    """
    real = real_df.copy()
    synth = synth_df.copy()
    for col in real.columns:
        if col not in synth.columns:
            continue
        r_num = pd.api.types.is_numeric_dtype(real[col])
        s_num = pd.api.types.is_numeric_dtype(synth[col])
        if r_num == s_num:
            continue
        # One is numeric, one is object -> coerce object side to numeric
        if not r_num:
            real[col] = pd.to_numeric(real[col], errors="coerce")
        if not s_num:
            synth[col] = pd.to_numeric(synth[col], errors="coerce")
    return real, synth


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run MIA (distinguishability) evaluation on ALM vs RealTabFormer synthetic data.")
    parser.add_argument(
        "--alm-paper-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing original and synthetic CSVs (default: ALM_Paper)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write JSON results (default: <alm-paper-dir>/evaluation_reports)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Max samples per dataset for speed (default: use all)",
    )
    args = parser.parse_args()
    base_dir = args.alm_paper_dir
    out_dir = args.out_dir or (base_dir / "evaluation_reports")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset config: (dataset_key, original_csv, [(method_name, synthetic_csv), ...])
    # All 11 ALM_Paper datasets for comparable MIA reports.
    datasets = [
        (
            "adult_census_income",
            base_dir / "Adult Census Income_original.csv",
            [
                ("alm", base_dir / "Adult Census Income_synthetic_alm.csv"),
                ("realtabformer", base_dir / "Adult Census Income_synthetic_rtf .csv"),  # note space before .csv
            ],
        ),
        (
            "diabetes_health_indicators",
            base_dir / "diabetes_health_indicators_original.csv",
            [
                ("alm", base_dir / "diabetes_health_indicators_synthetic_alm.csv"),
                ("realtabformer", base_dir / "diabetes_health_indicators_synthetic_rtf.csv"),
            ],
        ),
        (
            "breast_cancer",
            base_dir / "breast-cancer_original.csv",
            [
                ("alm", base_dir / "breast-cancer_original_synthetic_alm.csv"),
                ("realtabformer", base_dir / "breast-cancer_original_synthetic_rtf.csv"),
            ],
        ),
        (
            "parkinsons",
            base_dir / "parkinsons.csv",
            [
                ("alm", base_dir / "synthetic_alm_parkinsons.csv"),
                ("realtabformer", base_dir / "synthetic_rtf_parkinsons.csv"),
            ],
        ),
        (
            "obesity",
            base_dir / "Obesity.csv",
            [
                ("alm", base_dir / "synthetic_alm_Obesity.csv"),
                ("realtabformer", base_dir / "synthetic_rtf_Obesity.csv"),
            ],
        ),
        (
            "vn_banking",
            base_dir / "vn_banking_original.csv",
            [
                ("alm", base_dir / "vn_banking_synthetic_alm.csv"),
                ("realtabformer", base_dir / "vn_banking_synthetic_rtf.csv"),
            ],
        ),
        (
            "lung_cancer",
            base_dir / "lung_cancer.csv",
            [
                ("alm", base_dir / "synthetic_alm_lung_cancer.csv"),
                ("realtabformer", base_dir / "synthetic_rtf_lung_cancer.csv"),
            ],
        ),
        (
            "hypothyroid",
            base_dir / "hypothyroid.csv",
            [
                ("alm", base_dir / "synthetic_alm_hypothyroid.csv"),
                ("realtabformer", base_dir / "synthetic_rtf_hypothyroid.csv"),
            ],
        ),
        (
            "liver_disorders",
            base_dir / "liver_disorders.csv",
            [
                ("alm", base_dir / "synthetic_alm_liver_disorders.csv"),
                ("realtabformer", base_dir / "synthetic_rtf_liver_disorders.csv"),
            ],
        ),
        (
            "heart_failure_clinical_records",
            base_dir / "heart_failure_clinical_records_dataset.csv",
            [
                ("alm", base_dir / "synthetic_alm_heart_failure_clinical_records_dataset.csv"),
                ("realtabformer", base_dir / "synthetic_rtf_heart_failure_clinical_records_dataset.csv"),
            ],
        ),
        (
            "pir_vision_office",
            base_dir / "PIR Vision Office_original.csv",
            [
                ("alm", base_dir / "PIR Vision Office_synthetic_alm.csv"),
                ("realtabformer", base_dir / "PIR Vision Office_synthetic_rtf.csv"),
            ],
        ),
    ]

    all_results = {}
    for dataset_key, original_path, methods in datasets:
        if not original_path.exists():
            logger.warning("Skipping %s: original not found at %s", dataset_key, original_path)
            continue
        real_df = pd.read_csv(original_path)
        logger.info("Dataset %s: real shape %s", dataset_key, real_df.shape)
        all_results[dataset_key] = {}
        for method_name, synthetic_path in methods:
            if not synthetic_path.exists():
                logger.warning("Skipping %s/%s: synthetic not found at %s", dataset_key, method_name, synthetic_path)
                continue
            synth_df = pd.read_csv(synthetic_path)
            # Align columns (fixes vn_banking, breast_cancer: drop IDs, rename Target -> is_churned/Class)
            try:
                real_aligned, synth_aligned = align_real_synthetic_columns(real_df, synth_df)
            except ValueError as e:
                logger.warning("Skipping %s/%s: column alignment failed: %s", dataset_key, method_name, e)
                continue
            # Normalize boolean-like columns (fixes hypothyroid: t/f vs 0/1 -> same encoding)
            real_aligned, synth_aligned = normalize_boolean_like_columns(real_aligned, synth_aligned)
            # Coerce numeric columns when one side is object (fixes hypothyroid: age string vs float)
            real_aligned, synth_aligned = coerce_numeric_columns(real_aligned, synth_aligned)
            # Drop target column so MIA measures feature-based distinguishability, not trivial
            # separation from synthetic target imbalance (e.g. vn_banking: real ~35%% churn vs synthetic ~1%%).
            target_col = DATASET_TARGET_COLUMN.get(dataset_key)
            if target_col and target_col in real_aligned.columns and target_col in synth_aligned.columns:
                real_aligned = real_aligned.drop(columns=[target_col])
                synth_aligned = synth_aligned.drop(columns=[target_col])
                logger.info("Dropped target column %s for MIA (feature-only distinguishability)", target_col)
            logger.info("Running MIA for %s (%s): aligned shapes real=%s synth=%s", dataset_key, method_name, real_aligned.shape, synth_aligned.shape)
            evaluator = MIAEvaluator(real_data=real_aligned, synthetic_data=synth_aligned, sample_size=args.sample_size)
            results = evaluator.evaluate()
            # Ensure full JSON serializability
            results_serializable = {
                "summary_metrics": results["summary_metrics"],
                "bootstrap_ci": results["bootstrap_ci"],
                "feature_importance": results["feature_importance"],
                "stratified_analysis": _json_safe(results["stratified_analysis"]),
            }
            all_results[dataset_key][method_name] = results_serializable
            out_file = out_dir / f"mia_{dataset_key}_{method_name}.json"
            with open(out_file, "w") as f:
                json.dump(results_serializable, f, indent=2)
            logger.info("Wrote %s", out_file)
            sm = results["summary_metrics"]
            print(f"\n[{dataset_key} | {method_name}] Overall distinguishability (AUC): {sm['overall_distinguishability_score']:.4f} "
                  f"(BB={sm['black_box_auc']:.4f}, LR={sm['logistic_regression_auc']:.4f}, WB={sm['white_box_auc']:.4f})")

    summary_path = out_dir / "mia_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Wrote summary %s", summary_path)
    print("\nDone. Summary written to", summary_path)