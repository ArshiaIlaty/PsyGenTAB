"""
Comprehensive Evaluation Script for All Synthetic Datasets
Based on balanced_comparison_evaluation.py methodology
Evaluates synthetic data quality using:
1. Statistical comparison
2. Machine learning utility (downstream tasks)
3. Privacy metrics
4. Distribution analysis
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
import sys
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDatasetEvaluator:
    """Comprehensive evaluator for synthetic vs real data"""
    
    def __init__(self, dataset_name, real_csv, synthetic_csv, target_column):
        self.dataset_name = dataset_name
        self.real_csv = real_csv
        self.synthetic_csv = synthetic_csv
        self.target_column = target_column
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        self.results_dir = f'evaluation_results/{dataset_name}'
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Initialized evaluator for {dataset_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_data(self):
        """Load real and synthetic datasets"""
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING: {self.dataset_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            self.real_df = pd.read_csv(self.real_csv)
            self.synthetic_df = pd.read_csv(self.synthetic_csv)
            
            logger.info(f"✓ Real data loaded: {self.real_df.shape}")
            logger.info(f"✓ Synthetic data loaded: {self.synthetic_df.shape}")
            
            # Display class distribution if target exists
            if self.target_column in self.real_df.columns:
                real_dist = self.real_df[self.target_column].value_counts()
                logger.info(f"Real data class distribution:\n{real_dist}")
                
                if self.target_column in self.synthetic_df.columns:
                    synth_dist = self.synthetic_df[self.target_column].value_counts()
                    logger.info(f"Synthetic data class distribution:\n{synth_dist}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def statistical_comparison(self):
        """Compare statistical properties between real and synthetic data"""
        logger.info(f"\n{'='*60}")
        logger.info("STATISTICAL COMPARISON")
        logger.info(f"{'='*60}")
        
        stats_results = {}
        
        # 1. Basic statistics
        numerical_cols = self.real_df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_column]
        
        logger.info(f"\nAnalyzing {len(numerical_cols)} numerical features...")
        
        for col in numerical_cols:
            if col in self.synthetic_df.columns:
                real_data = self.real_df[col].dropna()
                synthetic_data = self.synthetic_df[col].dropna()
                
                # Calculate statistics
                stats_results[col] = {
                    'real_mean': float(real_data.mean()),
                    'synthetic_mean': float(synthetic_data.mean()),
                    'real_std': float(real_data.std()),
                    'synthetic_std': float(synthetic_data.std()),
                    'real_median': float(real_data.median()),
                    'synthetic_median': float(synthetic_data.median()),
                    'mean_difference': abs(float(real_data.mean() - synthetic_data.mean())),
                    'std_difference': abs(float(real_data.std() - synthetic_data.std()))
                }
                
                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_pval = stats.ks_2samp(real_data, synthetic_data)
                    stats_results[col]['ks_statistic'] = float(ks_stat)
                    stats_results[col]['ks_pvalue'] = float(ks_pval)
                except:
                    pass
        
        # Calculate overall statistics
        mean_diffs = [stats_results[col]['mean_difference'] for col in stats_results]
        avg_mean_diff = np.mean(mean_diffs) if mean_diffs else 0
        
        logger.info(f"\nAverage mean difference across features: {avg_mean_diff:.4f}")
        
        # 2. Categorical features comparison
        categorical_cols = self.real_df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target_column]
        
        logger.info(f"\nAnalyzing {len(categorical_cols)} categorical features...")
        
        cat_stats = {}
        for col in categorical_cols:
            if col in self.synthetic_df.columns:
                real_dist = self.real_df[col].value_counts(normalize=True).to_dict()
                synth_dist = self.synthetic_df[col].value_counts(normalize=True).to_dict()
                
                cat_stats[col] = {
                    'real_distribution': real_dist,
                    'synthetic_distribution': synth_dist
                }
        
        self.stats_results = {'numerical': stats_results, 'categorical': cat_stats}
        return stats_results
    
    def preprocess_datasets_aligned(self, real_df, synth_df):
        """
        Preprocess real and synthetic datasets with aligned encoding.
        This ensures LabelEncoder uses the same vocabulary for both datasets.
        """
        # Drop problematic columns
        columns_to_drop = ['Unnamed: 32', 'id']
        for col in columns_to_drop:
            if col in real_df.columns:
                real_df = real_df.drop(columns=[col])
            if col in synth_df.columns:
                synth_df = synth_df.drop(columns=[col])
        
        # Separate features and target
        if self.target_column not in real_df.columns:
            logger.warning(f"Target column {self.target_column} not found in real dataset")
            return None, None, None, None
        
        X_real = real_df.drop(columns=[self.target_column])
        y_real = real_df[self.target_column]
        
        if self.target_column in synth_df.columns:
            X_synth = synth_df.drop(columns=[self.target_column])
            y_synth = synth_df[self.target_column]
        else:
            X_synth = synth_df.copy()
            y_synth = None
        
        # Find global categorical columns and create aligned encoders
        categorical_cols = X_real.select_dtypes(include=['object']).columns.tolist()
        encoders = {}
        
        if len(categorical_cols) > 0:
            # Combine datasets to find global vocabulary
            combined_X = pd.concat([X_real, X_synth], axis=0, ignore_index=True)
            
            for col in categorical_cols:
                if col in combined_X.columns:
                    # Fill NaN with mode
                    if combined_X[col].isna().any():
                        mode_val = combined_X[col].mode()[0] if len(combined_X[col].mode()) > 0 else 'Unknown'
                        combined_X[col] = combined_X[col].fillna(mode_val)
                    
                    # Convert to string to handle mixed types safely
                    combined_X[col] = combined_X[col].astype(str)
                    
                    # Fit encoder on combined vocabulary
                    le = LabelEncoder()
                    le.fit(combined_X[col])
                    encoders[col] = le
            
            # Transform separately using the SAME encoders
            def transform_with_encoders(df, encoders):
                df_copy = df.copy()
                for col, le in encoders.items():
                    if col in df_copy.columns:
                        # Fill NaN
                        if df_copy[col].isna().any():
                            mode_val = df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else le.classes_[0]
                            df_copy[col] = df_copy[col].fillna(mode_val)
                        
                        # Convert to string and handle unseen labels
                        df_copy[col] = df_copy[col].astype(str)
                        # Map unseen labels to first known class (or could use mode)
                        df_copy[col] = df_copy[col].apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        df_copy[col] = le.transform(df_copy[col])
                return df_copy
            
            X_real = transform_with_encoders(X_real, encoders)
            X_synth = transform_with_encoders(X_synth, encoders)
        
        # Handle numeric columns - fill NaN with median
        numeric_cols = X_real.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in X_real.columns:
                median_val = X_real[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_real[col] = X_real[col].fillna(median_val)
            if col in X_synth.columns:
                median_val = X_real[col].median()  # Use real median for consistency
                if pd.isna(median_val):
                    median_val = 0
                X_synth[col] = X_synth[col].fillna(median_val)
        
        return X_real, y_real, X_synth, y_synth
    
    def preprocess_data(self, df, is_real=True):
        """
        Legacy method for backward compatibility.
        WARNING: This method fits encoders separately and may cause label mismatch.
        Use preprocess_datasets_aligned() for proper evaluation.
        """
        df = df.copy()
        
        # Drop problematic columns
        columns_to_drop = ['Unnamed: 32', 'id']  # Drop ID and unnamed columns
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"Dropped column: {col}")
        
        # Separate features and target
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            logger.warning(f"Target column {self.target_column} not found in dataset")
            return None, None
        
        # Handle categorical variables BEFORE filling NaNs
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if col in X.columns:
                    # Fill NaN in categorical columns with mode
                    if X[col].isna().any():
                        mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                        X[col] = X[col].fillna(mode_val)
                    
                    le = LabelEncoder()
                    # Handle unseen values
                    unique_vals = X[col].dropna().unique()
                    le.fit(unique_vals)
                    X[col] = le.transform(X[col])
        
        # Handle missing values in numerical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numerical_cols) > 0:
            for col in numerical_cols:
                if X[col].isna().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X[col] = X[col].fillna(median_val)
        
        # Final check and fill any remaining NaNs with 0
        if X.isna().any().any():
            logger.warning(f"Still have NaNs after preprocessing, filling with 0")
            X = X.fillna(0)
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def machine_learning_utility(self):
        """Evaluate ML utility using multiple models and balancing techniques"""
        logger.info(f"\n{'='*60}")
        logger.info("MACHINE LEARNING UTILITY EVALUATION")
        logger.info(f"{'='*60}")
        
        # Preprocess data with aligned encoding (CRITICAL FIX for label encoding trap)
        try:
            X_real, y_real, X_synthetic, y_synthetic = self.preprocess_datasets_aligned(
                self.real_df, self.synthetic_df
            )
            if X_real is None:
                logger.error("Failed to preprocess data")
                return {}
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
        
        # Ensure both datasets have the same columns in the same order
        # Find common columns (intersection)
        common_cols = list(set(X_real.columns) & set(X_synthetic.columns))
        if len(common_cols) != len(X_real.columns) or len(common_cols) != len(X_synthetic.columns):
            logger.warning(f"Column mismatch: Real has {len(X_real.columns)} cols, Synthetic has {len(X_synthetic.columns)} cols, Common: {len(common_cols)}")
            logger.warning(f"Real only: {set(X_real.columns) - set(X_synthetic.columns)}")
            logger.warning(f"Synthetic only: {set(X_synthetic.columns) - set(X_real.columns)}")
            # Use only common columns, sorted to ensure consistent order
            common_cols = sorted(common_cols)
            X_real = X_real[common_cols]
            X_synthetic = X_synthetic[common_cols]
        
        # Ensure column order matches (use real data column order as reference)
        X_synthetic = X_synthetic[X_real.columns]
        
        # Split real data for testing
        # Check if stratification is possible (each class needs at least 2 samples)
        try:
            unique, counts = np.unique(y_real, return_counts=True)
            min_class_count = counts.min()
            can_stratify = min_class_count >= 2
        except:
            can_stratify = False
        
        if can_stratify:
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
            )
        else:
            logger.warning("Cannot stratify real data split (some classes have < 2 samples), using non-stratified split")
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, test_size=0.3, random_state=42
            )
        
        logger.info(f"\nReal data split:")
        logger.info(f"  Train: {X_train_real.shape}, Test: {X_test_real.shape}")
        logger.info(f"  Synthetic training data: {X_synthetic.shape}")
        
        # Define models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Define balancing techniques
        balancing_techniques = ['none', 'class_weight', 'smote', 'undersample']
        
        results = {}
        
        # Scenario 1: Train on Real, Test on Real (Baseline)
        logger.info(f"\n--- Scenario 1: Train on Real, Test on Real (Baseline) ---")
        results['real_to_real'] = {}
        
        for technique in balancing_techniques:
            logger.info(f"\nBalancing: {technique}")
            results['real_to_real'][technique] = self.evaluate_models(
                X_train_real, y_train_real, X_test_real, y_test_real,
                models, technique
            )
        
        # Scenario 2: Train on Synthetic, Test on Real
        logger.info(f"\n--- Scenario 2: Train on Synthetic, Test on Real ---")
        results['synthetic_to_real'] = {}
        
        # Ensure test set has same column order as training set
        X_test_real_aligned = X_test_real[X_synthetic.columns]
        
        for technique in balancing_techniques:
            logger.info(f"\nBalancing: {technique}")
            results['synthetic_to_real'][technique] = self.evaluate_models(
                X_synthetic, y_synthetic, X_test_real_aligned, y_test_real,
                models, technique
            )
        
        # Scenario 3: Train on Real, Test on Synthetic
        logger.info(f"\n--- Scenario 3: Train on Real, Test on Synthetic ---")
        # Check if stratification is possible for synthetic data
        try:
            unique, counts = np.unique(y_synthetic, return_counts=True)
            min_class_count = counts.min()
            can_stratify = min_class_count >= 2
        except:
            can_stratify = False
        
        if can_stratify:
            X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
                X_synthetic, y_synthetic, test_size=0.3, random_state=42, stratify=y_synthetic
            )
        else:
            logger.warning("Cannot stratify synthetic data split (some classes have < 2 samples), using non-stratified split")
            X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
                X_synthetic, y_synthetic, test_size=0.3, random_state=42
            )
        
        # Ensure test set has same column order as training set
        X_test_synth = X_test_synth[X_train_real.columns]
        
        results['real_to_synthetic'] = {}
        for technique in balancing_techniques:
            logger.info(f"\nBalancing: {technique}")
            results['real_to_synthetic'][technique] = self.evaluate_models(
                X_train_real, y_train_real, X_test_synth, y_test_synth,
                models, technique
            )
        
        self.ml_results = results
        return results
    
    def evaluate_models(self, X_train, y_train, X_test, y_test, models, balancing='none'):
        """Evaluate models with optional balancing"""
        
        # Apply balancing
        if balancing == 'class_weight':
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            weight_dict = dict(zip(np.unique(y_train), class_weights))
            models_with_weights = {
                'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight=weight_dict),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weight_dict),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight=weight_dict)
            }
            models = models_with_weights
        elif balancing == 'smote':
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train)-1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"  SMOTE applied: training size = {X_train.shape[0]}")
            except Exception as e:
                logger.warning(f"  SMOTE failed: {e}, using original data")
        elif balancing == 'undersample':
            try:
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                logger.info(f"  Undersampling applied: training size = {X_train.shape[0]}")
            except Exception as e:
                logger.warning(f"  Undersampling failed: {e}, using original data")
        
        results = {}
        
        # Align columns between train and test before model training
        # This ensures feature order matches for sklearn models
        if hasattr(X_train, 'columns') and hasattr(X_test, 'columns'):
            # Get intersection of columns
            common_cols = [col for col in X_train.columns if col in X_test.columns]
            if len(common_cols) != len(X_train.columns) or len(common_cols) != len(X_test.columns):
                logger.warning(f"Column mismatch: Train has {len(X_train.columns)} cols, Test has {len(X_test.columns)} cols, Using {len(common_cols)} common cols")
                missing_train = set(X_train.columns) - set(X_test.columns)
                missing_test = set(X_test.columns) - set(X_train.columns)
                if missing_train:
                    logger.warning(f"  Columns in train but not in test: {missing_train}")
                if missing_test:
                    logger.warning(f"  Columns in test but not in train: {missing_test}")
            # Use only common columns in the order they appear in X_train
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'kappa': cohen_kappa_score(y_test, y_pred)
                }
                
                # ROC AUC - supports both binary and multiclass
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_test)
                        n_classes = len(np.unique(y_test))
                        
                        if n_classes == 2:
                            # Binary classification - use probability of positive class
                            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                        elif n_classes > 2:
                            # Multiclass classification - use one-vs-rest with macro average
                            # Check if all classes are present in test set
                            if len(np.unique(y_test)) == n_classes:
                                metrics['roc_auc'] = roc_auc_score(
                                    y_test, y_pred_proba, 
                                    multi_class='ovr', 
                                    average='macro'
                                )
                            else:
                                # Some classes missing in test set - use weighted average
                                metrics['roc_auc'] = roc_auc_score(
                                    y_test, y_pred_proba,
                                    multi_class='ovr',
                                    average='weighted'
                                )
                        else:
                            # Only one class in test set - cannot calculate ROC AUC
                            metrics['roc_auc'] = None
                    except Exception as e:
                        # If ROC AUC calculation fails, set to None
                        logger.debug(f"  ROC AUC calculation failed: {e}")
                        metrics['roc_auc'] = None
                else:
                    metrics['roc_auc'] = None
                
                results[model_name] = metrics
                
                logger.info(f"  {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, Balanced Acc={metrics['balanced_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"  Error evaluating {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def privacy_evaluation(self):
        """Evaluate privacy metrics"""
        logger.info(f"\n{'='*60}")
        logger.info("PRIVACY EVALUATION")
        logger.info(f"{'='*60}")
        
        privacy_metrics = {}
        
        # 1. Exact duplication check
        real_str = self.real_df.astype(str).apply(lambda x: '|'.join(x), axis=1)
        synthetic_str = self.synthetic_df.astype(str).apply(lambda x: '|'.join(x), axis=1)
        
        exact_matches = len(set(real_str) & set(synthetic_str))
        privacy_metrics['exact_matches'] = exact_matches
        privacy_metrics['exact_match_ratio'] = exact_matches / len(self.synthetic_df)
        
        logger.info(f"\nExact matches with real data: {exact_matches} ({privacy_metrics['exact_match_ratio']:.4f})")
        
        # 2. Uniqueness
        real_unique = len(self.real_df.drop_duplicates())
        synthetic_unique = len(self.synthetic_df.drop_duplicates())
        
        privacy_metrics['real_uniqueness'] = real_unique / len(self.real_df)
        privacy_metrics['synthetic_uniqueness'] = synthetic_unique / len(self.synthetic_df)
        
        logger.info(f"Real data uniqueness: {privacy_metrics['real_uniqueness']:.4f}")
        logger.info(f"Synthetic data uniqueness: {privacy_metrics['synthetic_uniqueness']:.4f}")
        
        # 3. Distance to Closest Record (DCR) - Vectorized for full dataset
        # CRITICAL FIX: Use full dataset with vectorized operations instead of sampling
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.preprocessing import StandardScaler
        
        # Use numeric columns only
        real_num = self.real_df.select_dtypes(include=[np.number]).fillna(0)
        synth_num = self.synthetic_df.select_dtypes(include=[np.number]).fillna(0)
        
        # Align columns
        common_cols = list(set(real_num.columns) & set(synth_num.columns))
        if len(common_cols) == 0:
            logger.warning("No common numeric columns for DCR calculation")
            privacy_metrics['avg_dcr'] = None
            privacy_metrics['min_dcr'] = None
            privacy_metrics['5th_percentile_dcr'] = None
        else:
            real_num = real_num[common_cols]
            synth_num = synth_num[common_cols]
            
            # Standardize for fair distance calculation
            scaler = StandardScaler()
            real_num_scaled = scaler.fit_transform(real_num)
            synth_num_scaled = scaler.transform(synth_num)
            
            # Compute distances in batches if dataset is huge (>20k rows)
            # For typical medical datasets (<10k), this is instant
            if len(synth_num) > 20000:
                # Batch processing for very large datasets
                batch_size = 5000
                min_dists = []
                logger.info(f"Computing DCR in batches (synthetic size: {len(synth_num)})...")
                
                for i in range(0, len(synth_num_scaled), batch_size):
                    batch_end = min(i + batch_size, len(synth_num_scaled))
                    batch_synth = synth_num_scaled[i:batch_end]
                    dists = euclidean_distances(batch_synth, real_num_scaled)
                    batch_min_dists = dists.min(axis=1)
                    min_dists.extend(batch_min_dists)
                    logger.info(f"  Processed batch {i//batch_size + 1}/{(len(synth_num_scaled)-1)//batch_size + 1}")
                
                min_dists = np.array(min_dists)
            else:
                # Full matrix computation for smaller datasets
                logger.info(f"Computing DCR for full dataset (synthetic size: {len(synth_num)})...")
                dists = euclidean_distances(synth_num_scaled, real_num_scaled)
                min_dists = dists.min(axis=1)  # Min distance for each synthetic record to ANY real record
            
            privacy_metrics['avg_dcr'] = float(np.mean(min_dists))
            privacy_metrics['min_dcr'] = float(np.min(min_dists))  # Crucial: Is any record TOO close?
            privacy_metrics['5th_percentile_dcr'] = float(np.percentile(min_dists, 5))  # Risk threshold
            privacy_metrics['median_dcr'] = float(np.median(min_dists))
            privacy_metrics['95th_percentile_dcr'] = float(np.percentile(min_dists, 95))
            
            logger.info(f"Average DCR (N={len(synth_num)}): {privacy_metrics['avg_dcr']:.4f}")
            logger.info(f"Minimum DCR: {privacy_metrics['min_dcr']:.4f}")
            logger.info(f"5th Percentile DCR: {privacy_metrics['5th_percentile_dcr']:.4f}")
        
        self.privacy_metrics = privacy_metrics
        return privacy_metrics
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info(f"\n{'='*60}")
        logger.info("GENERATING VISUALIZATIONS")
        logger.info(f"{'='*60}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. ML Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        self.plot_ml_comparison(ax1)
        
        # 2. Distribution comparison for key features
        numerical_cols = self.real_df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_column and col in self.synthetic_df.columns]
        
        if len(numerical_cols) >= 2:
            ax2 = plt.subplot(2, 3, 2)
            self.plot_distribution_comparison(ax2, numerical_cols[0])
            
            ax3 = plt.subplot(2, 3, 3)
            self.plot_distribution_comparison(ax3, numerical_cols[1])
        
        # 3. Privacy metrics
        ax4 = plt.subplot(2, 3, 4)
        self.plot_privacy_metrics(ax4)
        
        # 4. Statistical comparison
        ax5 = plt.subplot(2, 3, 5)
        self.plot_statistical_comparison(ax5)
        
        # 5. Class distribution comparison
        ax6 = plt.subplot(2, 3, 6)
        self.plot_class_distribution(ax6)
        
        plt.tight_layout()
        viz_path = os.path.join(self.results_dir, f'{self.dataset_name}_comprehensive_evaluation.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_path}")
    
    def plot_ml_comparison(self, ax):
        """Plot ML performance comparison"""
        if not hasattr(self, 'ml_results') or self.ml_results is None:
            ax.text(0.5, 0.5, 'No ML Results Available', ha='center', va='center')
            return
        
        scenarios = ['real_to_real', 'synthetic_to_real']
        scenario_labels = ['Real→Real\n(Baseline)', 'Synthetic→Real']
        
        # Get balanced accuracy for Random Forest with no balancing
        scores = []
        for scenario in scenarios:
            score = 0  # Default score
            try:
                if (scenario in self.ml_results and 
                    self.ml_results[scenario] is not None and 
                    'none' in self.ml_results[scenario] and
                    self.ml_results[scenario]['none'] is not None and
                    'Random Forest' in self.ml_results[scenario]['none'] and
                    self.ml_results[scenario]['none']['Random Forest'] is not None):
                    score = self.ml_results[scenario]['none']['Random Forest'].get('balanced_accuracy', 0)
            except (TypeError, KeyError, AttributeError) as e:
                logger.warning(f"Could not get score for {scenario}: {e}")
            scores.append(score)
        
        bars = ax.bar(scenario_labels, scores, alpha=0.8, color=['#2E8B57', '#FF6B6B'])
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('ML Performance: Random Forest')
        ax.set_ylim([0, 1])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_distribution_comparison(self, ax, column):
        """Plot distribution comparison for a feature"""
        if column in self.real_df.columns and column in self.synthetic_df.columns:
            real_data = self.real_df[column].dropna()
            synthetic_data = self.synthetic_df[column].dropna()
            
            ax.hist(real_data, bins=30, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(synthetic_data, bins=30, alpha=0.5, label='Synthetic', color='red', density=True)
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution: {column}')
            ax.legend()
    
    def plot_privacy_metrics(self, ax):
        """Plot privacy metrics"""
        if not hasattr(self, 'privacy_metrics'):
            return
        
        metrics = ['exact_match_ratio', 'synthetic_uniqueness']
        labels = ['Exact Match\nRatio', 'Synthetic\nUniqueness']
        values = [self.privacy_metrics.get(m, 0) for m in metrics]
        
        bars = ax.bar(labels, values, alpha=0.8, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Ratio')
        ax.set_title('Privacy Metrics')
        ax.set_ylim([0, 1])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_statistical_comparison(self, ax):
        """Plot statistical comparison"""
        if not hasattr(self, 'stats_results'):
            return
        
        if 'numerical' in self.stats_results:
            features = list(self.stats_results['numerical'].keys())[:5]  # Top 5 features
            mean_diffs = [self.stats_results['numerical'][f]['mean_difference'] for f in features]
            
            bars = ax.barh(features, mean_diffs, alpha=0.8, color='orange')
            ax.set_xlabel('Mean Difference')
            ax.set_title('Statistical Differences')
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center')
    
    def plot_class_distribution(self, ax):
        """Plot class distribution comparison"""
        if self.target_column in self.real_df.columns and self.target_column in self.synthetic_df.columns:
            real_dist = self.real_df[self.target_column].value_counts(normalize=True)
            synth_dist = self.synthetic_df[self.target_column].value_counts(normalize=True)
            
            x = np.arange(len(real_dist))
            width = 0.35
            
            ax.bar(x - width/2, real_dist.values, width, label='Real', alpha=0.8, color='blue')
            ax.bar(x + width/2, synth_dist.values, width, label='Synthetic', alpha=0.8, color='red')
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Proportion')
            ax.set_title(f'Class Distribution: {self.target_column}')
            ax.set_xticks(x)
            ax.set_xticklabels(real_dist.index)
            ax.legend()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        
        # Run all evaluations
        if not self.load_data():
            return None
        
        self.statistical_comparison()
        self.machine_learning_utility()
        self.privacy_evaluation()
        self.generate_visualizations()
        
        # Generate text report
        report_path = os.path.join(self.results_dir, f'{self.dataset_name}_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"COMPREHENSIVE EVALUATION REPORT\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            # Dataset info
            f.write(f"DATASET INFORMATION\n")
            f.write(f"Real data shape: {self.real_df.shape}\n")
            f.write(f"Synthetic data shape: {self.synthetic_df.shape}\n\n")
            
            # ML Results Summary
            f.write(f"MACHINE LEARNING UTILITY\n")
            f.write(f"-" * 60 + "\n")
            if hasattr(self, 'ml_results'):
                for scenario, techniques in self.ml_results.items():
                    f.write(f"\n{scenario.upper()}:\n")
                    for technique, models in techniques.items():
                        f.write(f"  {technique}:\n")
                        for model, metrics in models.items():
                            if metrics:
                                f.write(f"    {model}: ")
                                f.write(f"Accuracy={metrics['accuracy']:.4f}, ")
                                f.write(f"F1={metrics['f1_score']:.4f}, ")
                                f.write(f"Balanced Acc={metrics['balanced_accuracy']:.4f}\n")
            
            # Privacy Metrics
            f.write(f"\n\nPRIVACY METRICS\n")
            f.write(f"-" * 60 + "\n")
            if hasattr(self, 'privacy_metrics'):
                for metric, value in self.privacy_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
        
        logger.info(f"Report saved to {report_path}")
        logger.info(f"Results directory: {self.results_dir}")
        
        return report_path

def evaluate_all_datasets():
    """Evaluate all generated synthetic datasets"""
    
    datasets = [
        # Primary datasets with hierarchical discriminators
        {
            'name': 'diabetes',
            'real': '../diabetes.csv',
            'synthetic': '../output_hierarchical_diabetes_clean.csv',
            'target': 'diabetes'
        },
        {
            'name': 'heloc',
            'real': '../heloc.csv',
            'synthetic': '../output_hierarchical_heloc_clean.csv',
            'target': 'RiskPerformance'
        },
        # Medical datasets
        {
            'name': 'breast_cancer',
            'real': 'breast_cancer/breast-data.csv',
            'synthetic': 'synthetic_breast-data_clean.csv',  # Use cleaned version
            'target': 'diagnosis'
        },
        {
            'name': 'heart_failure',
            'real': 'heart_failure/heart_failure_clinical_records_dataset.csv',
            'synthetic': 'synthetic_heart_failure_clinical_records_dataset.csv',
            'target': 'DEATH_EVENT'
        },
        {
            'name': 'hypothyroid',
            'real': 'hypothyroid/hypothyroid.csv',
            'synthetic': 'synthetic_hypothyroid_clean.csv',  # Use cleaned version
            'target': 'binaryClass'
        },
        {
            'name': 'obesity',
            'real': 'obesity/ObesityDataSet_raw_and_data_sinthetic.csv',
            'synthetic': 'synthetic_ObesityDataSet_raw_and_data_sinthetic.csv',
            'target': 'NObeyesdad'
        },
        # New UCI datasets
        {
            'name': 'parkinsons',
            'real': 'parkinsons/parkinsons.csv',
            'synthetic': 'synthetic_parkinsons.csv',
            'target': 'status'
        },
        {
            'name': 'german_credit',
            'real': 'german_credit/german_credit.csv',
            'synthetic': 'synthetic_german_credit.csv',
            'target': 'class'
        },
        {
            'name': 'liver_disorders',
            'real': 'liver_disorders/liver_disorders.csv',
            'synthetic': 'synthetic_liver_disorders.csv',
            'target': 'selector'
        },
        {
            'name': 'bank_marketing',
            'real': 'bank_marketing/bank_marketing.csv',
            'synthetic': 'synthetic_bank_marketing.csv',
            'target': 'y'
        }
        # Skipping census_income due to continuous target issue - needs data cleaning
    ]
    
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# EVALUATING: {dataset['name'].upper()}")
        logger.info(f"{'#'*80}")
        
        try:
            evaluator = ComprehensiveDatasetEvaluator(
                dataset_name=dataset['name'],
                real_csv=dataset['real'],
                synthetic_csv=dataset['synthetic'],
                target_column=dataset['target']
            )
            
            report_path = evaluator.generate_comprehensive_report()
            all_results[dataset['name']] = {
                'status': 'completed',
                'report_path': report_path
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset['name']}: {e}")
            all_results[dataset['name']] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Generate final summary
    logger.info(f"\n\n{'='*80}")
    logger.info("FINAL SUMMARY - ALL DATASETS")
    logger.info(f"{'='*80}")
    
    for dataset_name, result in all_results.items():
        status_emoji = "✅" if result['status'] == 'completed' else "❌"
        logger.info(f"{status_emoji} {dataset_name}: {result['status']}")
    
    logger.info(f"\n🎉 Evaluation complete! Check evaluation_results/ for detailed reports.")
    
    return all_results

if __name__ == "__main__":
    results = evaluate_all_datasets()
