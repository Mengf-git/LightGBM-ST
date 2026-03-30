import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import json
import shap

warnings.filterwarnings("ignore")

# ----------------------
# Unified Font Settings
# ----------------------
rcParams['font.family'] = ['Times New Roman','Arial']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12           # Base font size
rcParams['axes.labelsize'] = 12      # Axis label size
rcParams['axes.titlesize'] = 13      # Subplot title size
rcParams['legend.fontsize'] = 10     # Legend font size
rcParams['figure.titlesize'] = 14    # Figure title size
rcParams['xtick.labelsize'] = 10     # X-axis tick label size
rcParams['ytick.labelsize'] = 10     # Y-axis tick label size
rcParams['lines.linewidth'] = 1.8    # Line width

# =====================================================================
# Core Configuration Parameters
# =====================================================================
FOLDER = "D:/Grade 1/GNSS-LSTM+Attention+SG/Spatial Correlation - Machine Learning/YNYS"
TARGET_STATION = "YNYS"
NEIGHBOR_STATION = "YNLJ"
TIME_COL = "YYYYMMDD"
VALUE_COL = "U(m)"
START_DATE = "2016-03-28"
END_DATE = "2019-06-09"
N_REPEATS = 10  # Number of repetitions per experiment
RANDOM_SEED = 42
CORRELATION_R = 0.8503  # Obtained from spatial correlation analysis
NEIGHBOR_DIST_KM = 71.56  # Inter-station distance
BIAS_VALUE = -2.6672  # Systematic bias obtained from spatial analysis

# LightGBM Hyperparameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'n_estimators': 2000,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0,
    'reg_lambda': 1e-3,
    'random_state': RANDOM_SEED,
    'verbose': -1
}


# =====================================================================


class SpatialCorrelationFeatureEngineering:
    """Spatial correlation feature engineering class"""

    def __init__(self, correlation_r, neighbor_dist_km, bias_value):
        """
        Parameters:
            correlation_r: Pearson correlation coefficient
            neighbor_dist_km: Inter-station distance (km)
            bias_value: Systematic bias (target - neighbor)
        """
        self.correlation_r = correlation_r
        self.neighbor_dist_km = neighbor_dist_km
        self.bias_value = bias_value
        self.ols_model = None

    def fit_bias_correction(self, target_series, neighbor_series):
        """
        Fit OLS bias correction model on training set:
        U_target = a + b * U_neighbor
        """
        valid_mask = target_series.notna() & neighbor_series.notna()

        if valid_mask.sum() < 30:
            print("    [Warning] Insufficient training samples, using default parameters")
            self.ols_model = {'intercept': 0.0, 'coef': 1.0}
            return

        X = neighbor_series[valid_mask].values.reshape(-1, 1)
        y = target_series[valid_mask].values

        lr = LinearRegression()
        lr.fit(X, y)

        self.ols_model = {
            'intercept': lr.intercept_,
            'coef': lr.coef_[0]
        }

        print(f"    ✓ Bias correction: U_target = {lr.intercept_:.4f} + {lr.coef_[0]:.4f} * U_neighbor")

    def apply_bias_correction(self, neighbor_series):
        """Apply bias correction"""
        if self.ols_model is None:
            return neighbor_series.copy()

        return self.ols_model['intercept'] + self.ols_model['coef'] * neighbor_series

    def create_features(self, target_series, neighbor_series,
                        n_lags=7, rolling_windows=[3, 7, 14],
                        include_target_lags=True, use_neighbor_lag0=False):
        """
        Construct complete feature matrix

        Feature categories:
        1. Temporal features: doy_sin, doy_cos, month, year
        2. Neighbor station lags: nbr_lag0~lagN
        3. Neighbor station rolling statistics: rollmean, rollstd (multi-window)
        4. Target station history: tgt_lag1~lag3 (past values only)
        5. Spatial meta-features: neighbor_r, neighbor_dist_km
        6. Bias correction features: nbr_adj_lag0
        7. Missing indicators: nbr_isnan_lag0
        """
        """
        Construct reduced feature matrix (Scheme A: Conservative reduction)

        Removed features:
        - month_norm (redundant with doy_sin/cos)
        - nbr_lag4, nbr_lag5, nbr_lag7 (redundant long lags)
        - tgt_lag3, tgt_lag7 (redundant long lags)
        - nbr_rollmin_*, nbr_rollmax_* (redundant statistics)
        - 14-day window rolling statistics (redundant with 7-day window)

        Retained feature categories:
        1. Temporal features: doy_sin, doy_cos, year_norm
        2. Neighbor lags: nbr_lag0~lag3 (removed 4, 5, 7)
        3. Neighbor rolling statistics: rollmean, rollstd (3 and 7-day windows only, removed min/max)
        4. Target station history: tgt_lag1, tgt_lag2 (removed 3, 7)
        5. Spatial meta-features: neighbor_r, neighbor_dist_km, spatial_weight
        6. Dynamic spatial features: spatial_corr_*, nbr_tgt_diff_std_*
        7. Bias correction features: nbr_adj_lag0~lag2
        8. Missing indicators: nbr_isnan_lag0, n_neighbors_available, nbr_consecutive_valid
        """
        # Create DataFrame
        df = pd.DataFrame({
            'target': target_series,
            'neighbor': neighbor_series
        })

        # === 1. Temporal features ===
        df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        # df['month_norm'] = df.index.month / 12.0
        years = pd.Series(df.index.year)
        df['year_norm'] = (years - years.mean()) / years.std()

        # === 2. Neighbor station lag features (lag0 allowed for offline imputation) ===
        # (Restricted to lag0~lag3; lags 4, 5, 7 removed)
        max_useful_lag = 3
        for lag in range(n_lags + 1):
            if lag == 0 and not use_neighbor_lag0:
                continue
            df[f'nbr_lag{lag}'] = df['neighbor'].shift(lag)

        # === 3. Neighbor station rolling statistics (past window only) ===
        # Only 7-day window retained; min/max and 14-day window removed
        useful_windows = [w for w in rolling_windows if w in [3, 7]]
        for window in rolling_windows:
            # shift(1) ensures current value is excluded
            # Only mean and std retained; max/min removed
            df[f'nbr_rollmean_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).mean()
            df[f'nbr_rollstd_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).std().fillna(0)
            #df[f'nbr_rollmax_{window}'] = df['neighbor'].shift(1).rolling(
                #window, min_periods=1).max()
            #df[f'nbr_rollmin_{window}'] = df['neighbor'].shift(1).rolling(
                #window, min_periods=1).min()

        # === 4. Target station historical lags (past values only; only lag1, lag2 retained) ===
        if include_target_lags:
            for lag in [1, 2,]:
                df[f'tgt_lag{lag}'] = df['target'].shift(lag)

        # === 5. Spatial meta-features (static) ===
        df['neighbor_r'] = self.correlation_r
        df['neighbor_dist_km'] = self.neighbor_dist_km
        df['spatial_weight'] = self.correlation_r / (self.neighbor_dist_km + 1e-3)

        # === 5.5 Dynamic spatial consistency features ===
        # Rolling spatial correlation within time windows
        for window in [30, 60]:
            # Rolling correlation between target and neighbor stations
            rolling_corr = df['target'].rolling(window).corr(df['neighbor'])
            df[f'spatial_corr_{window}d'] = rolling_corr.shift(1)  # shift to avoid data leakage

            # Rolling std of difference between neighbor and target (captures systematic bias variation)
            diff = (df['neighbor'] - df['target']).shift(1)
            df[f'nbr_tgt_diff_std_{window}d'] = diff.rolling(window).std().fillna(0)

        # Consecutive valid days for neighbor station (assesses neighbor data quality)
        df['nbr_consecutive_valid'] = (~df['neighbor'].isna()).astype(int).groupby(
            (df['neighbor'].isna() != df['neighbor'].isna().shift()).cumsum()
        ).cumsum()

        # === 6. Bias correction features ===
        if self.ols_model is not None:
            nbr_adj = self.apply_bias_correction(df['neighbor'])
            for lag in range(min(3, n_lags + 1)):
                df[f'nbr_adj_lag{lag}'] = nbr_adj.shift(lag)

        # === 7. Missing value indicators ===
        df['nbr_isnan_lag0'] = df['neighbor'].isna().astype(int)
        df['n_neighbors_available'] = (~df['neighbor'].isna()).astype(int)

        return df

    @staticmethod
    def get_feature_groups():
        """
        Return feature group definitions (used for ablation studies)
        """
        return {
            'time': ['doy_sin', 'doy_cos', 'year_norm'],

            'target_history': ['tgt_lag1', 'tgt_lag2'],

            'neighbor_basic': [
                # Neighbor station lags
                'nbr_lag0', 'nbr_lag1', 'nbr_lag2', 'nbr_lag3',
                # Neighbor rolling statistics
                'nbr_rollmean_3', 'nbr_rollstd_3',
                'nbr_rollmean_7', 'nbr_rollstd_7',
                'nbr_rollmean_14', 'nbr_rollstd_14',
                # Missing indicators
                'nbr_isnan_lag0', 'n_neighbors_available',
                'nbr_consecutive_valid'
            ],

            'spatial_meta': [
                'neighbor_r', 'neighbor_dist_km', 'spatial_weight'
            ],

            'spatial_dynamic': [
                'spatial_corr_30d', 'spatial_corr_60d',
                'nbr_tgt_diff_std_30d', 'nbr_tgt_diff_std_60d'
            ],

            'bias_correction': [
                'nbr_adj_lag0', 'nbr_adj_lag1', 'nbr_adj_lag2'
            ]
        }

    @staticmethod
    def select_features_by_groups(df_features, group_names):
        """
        Select features by group names

        Parameters:
            df_features: Complete feature DataFrame
            group_names: list of str, names of feature groups to include

        Returns:
            selected_features: List of selected feature column names
        """
        feature_groups = SpatialCorrelationFeatureEngineering.get_feature_groups()
        selected = []

        for group_name in group_names:
            if group_name in feature_groups:
                selected.extend(feature_groups[group_name])

        # Retain only features that actually exist in the DataFrame
        existing_features = [f for f in selected if f in df_features.columns]

        # Always retain target and neighbor columns (for downstream processing)
        return existing_features


class LightGBMSpatialInterpolator:
    """LightGBM-based spatial correlation interpolator"""

    def __init__(self, feature_engineer, lgbm_params=None):
        self.feature_engineer = feature_engineer
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.model = None
        self.feature_names = None
        # ========== New: SHAP-related attributes ==========
        self.shap_explainer = None
        self.shap_values = None
        self.X_sample_for_shap = None

    def _prepare_train_data(self, df_features, train_idx):
        """Prepare training data"""
        train_df = df_features.loc[train_idx]

        # Retain only rows where target is non-NaN
        train_df = train_df.dropna(subset=['target'])

        if len(train_df) < 50:
            raise ValueError(f"Insufficient training samples: {len(train_df)}")

        # Separate features and target
        X = train_df.drop(columns=['target', 'neighbor'], errors='ignore')
        y = train_df['target']

        # Fill NaN in features with training set mean
        X = X.fillna(X.mean())

        self.feature_names = X.columns.tolist()

        return X, y

    def train(self, df_features, train_idx):
        """Train a single LightGBM model"""
        print(f"    Training LightGBM model...")

        # Prepare full training data
        X_full, y_full = self._prepare_train_data(df_features, train_idx)

        # Split into Train/Val (time series split)
        split_point = int(0.8 * len(X_full))
        X_train = X_full.iloc[:split_point]
        y_train = y_full.iloc[:split_point]
        X_val = X_full.iloc[split_point:]
        y_val = y_full.iloc[split_point:]

        # Train model
        self.model = lgb.LGBMRegressor(**self.lgbm_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100)]
        )

        print(f"    ✓ Model training complete")

    def predict(self, df_features, predict_idx):
        """Generate predictions"""
        X_pred = df_features.loc[predict_idx].drop(
            columns=['target', 'neighbor'], errors='ignore')

        # Fill NaN
        X_pred = X_pred.fillna(X_pred.mean())

        # Predict
        pred = self.model.predict(X_pred)

        return pred

    def get_feature_importance(self, top_n=15):
        """Retrieve feature importance"""
        if self.model is None:
            return None

        # Get feature importances
        importances = self.model.feature_importances_

        # Sort and return top_n
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    # ========== New: SHAP-related methods ==========

    def compute_shap_values(self, X_data, max_samples=1000):
        """
        Compute SHAP values

        Parameters:
            X_data: Feature matrix (DataFrame)
            max_samples: Maximum number of samples for SHAP computation (SHAP is computationally expensive)

        Returns:
            shap_values: SHAP value array
            X_sample: Sample used for computation
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first.")

        # Limit sample size to accelerate computation
        if len(X_data) > max_samples:
            print(f"    Sampling {max_samples}/{len(X_data)} samples for SHAP computation...")
            sample_indices = np.random.choice(len(X_data), max_samples, replace=False)
            X_sample = X_data.iloc[sample_indices].copy()
        else:
            X_sample = X_data.copy()

        print(f"    Computing SHAP values (sample size: {len(X_sample)})...")

        try:
            # Create SHAP explainer (TreeExplainer is suitable for tree-based models)
            self.shap_explainer = shap.TreeExplainer(self.model)

            # Compute SHAP values
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            self.X_sample_for_shap = X_sample

            print(f"    ✓ SHAP computation complete")

        except Exception as e:
            print(f"    [Warning] SHAP computation failed: {e}")
            self.shap_values = None
            self.X_sample_for_shap = None
            return None, None

        return self.shap_values, X_sample

    def get_shap_feature_importance(self, top_n=15):
        """
        Global feature importance based on SHAP values

        Returns:
            DataFrame: Feature importance ranking
        """
        if self.shap_values is None:
            raise ValueError("Please call compute_shap_values() first.")

        # Global importance = mean of |SHAP values|
        shap_importance = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False).head(top_n)

        return importance_df

    def compare_feature_importance_methods(self):
        """
        Compare LightGBM built-in and SHAP feature importance methods

        Returns:
            DataFrame: Importance values and rank comparison from both methods
        """
        if self.shap_values is None:
            raise ValueError("Please call compute_shap_values() first.")

        # 1. LightGBM built-in importance
        lgbm_importance = self.model.feature_importances_

        # 2. SHAP importance
        shap_importance = np.abs(self.shap_values).mean(axis=0)

        # Build comparison table
        comparison_df = pd.DataFrame({
            'feature': self.feature_names,
            'lgbm_importance': lgbm_importance,
            'shap_importance': shap_importance
        })

        # Normalize to [0, 1]
        comparison_df['lgbm_norm'] = (
                comparison_df['lgbm_importance'] / comparison_df['lgbm_importance'].sum()
        )
        comparison_df['shap_norm'] = (
                comparison_df['shap_importance'] / comparison_df['shap_importance'].sum()
        )

        # Compute ranks
        comparison_df['lgbm_rank'] = comparison_df['lgbm_importance'].rank(ascending=False)
        comparison_df['shap_rank'] = comparison_df['shap_importance'].rank(ascending=False)
        comparison_df['rank_diff'] = abs(comparison_df['lgbm_rank'] - comparison_df['shap_rank'])

        return comparison_df.sort_values('shap_importance', ascending=False)

    # ===============================================


class BaselineInterpolators:
    """Baseline interpolation methods"""

    @staticmethod
    def time_plus_history_only(df_features, train_idx, test_idx):
        """
        New: Temporal features + target station history (true no-spatial-information baseline).
        This is the key control group for evaluating the contribution of spatial information.
        """
        features = ['doy_sin', 'doy_cos', 'year_norm',
                    'tgt_lag1', 'tgt_lag2']

        train_df = df_features.loc[train_idx].dropna(subset=['target'])
        X_train = train_df[features].fillna(train_df[features].mean())
        y_train = train_df['target']

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        test_df = df_features.loc[test_idx]
        X_test = test_df[features].fillna(train_df[features].mean())
        pred = model.predict(X_test)

        return pred

    @staticmethod
    def linear_interpolation(series, missing_indices):
        """Linear interpolation baseline

        Parameters:
            series: pd.Series, series with missing values
            missing_indices: DatetimeIndex or list of integer positions
        """
        filled = series.copy()

        # Use .loc for DatetimeIndex; use .iloc for integer positional index
        if isinstance(missing_indices, pd.DatetimeIndex):
            filled.loc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.loc[missing_indices].values
        else:
            # Assume integer positional index
            filled.iloc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.iloc[missing_indices].values

    def create_continuous_missing(data, n_segments, days_per_segment, seed=None):
    """Create continuous missing segments (keep consistent with original code)"""
    if seed is not None:
        np.random.seed(seed)

    masked_data = data.copy()
    missing_indices = []
    n_total = len(data)
    used_ranges = []

    required_space = n_segments * (days_per_segment + 10)
    if required_space > n_total:
        n_segments = max(1, n_total // (days_per_segment + 10))

    segments_created = 0
    for _ in range(n_segments):
        for attempt in range(1000):
            start_idx = np.random.randint(0, max(1, n_total - days_per_segment))
            end_idx = start_idx + days_per_segment
            candidate = set(range(start_idx, end_idx))

            if all(candidate.isdisjoint(r) for r in used_ranges):
                min_distance = 10
                too_close = any(
                    (0 < min(candidate) - max(r) < min_distance) or
                    (0 < min(r) - max(candidate) < min_distance)
                    for r in used_ranges
                )

                if not too_close:
                    used_ranges.append(candidate)
                    missing_indices.extend(candidate)
                    segments_created += 1
                    break

    masked_data.iloc[missing_indices] = np.nan
    return masked_data, sorted(list(set(missing_indices))), segments_created


def create_random_clustered_missing(data, missing_ratio, min_segment=5, max_segment=30, seed=None):
    """Create random clustered missing data"""
    if seed is not None:
        np.random.seed(seed)

    n_total = len(data)
    n_missing = int(n_total * missing_ratio)
    masked_data = data.copy()
    missing_indices = []

    current_pos = 0
    max_attempts = 1000
    attempts = 0

    while len(missing_indices) < n_missing and current_pos < n_total and attempts < max_attempts:
        attempts += 1

        remaining = n_total - current_pos

        # Calculate jump distance (avoid randint parameter error)
        if remaining > 30:
            jump = np.random.randint(5, min(30, remaining // 2))
        elif remaining > 2:
            jump_max = max(2, remaining // 2)  # Ensure at least 2
            jump = np.random.randint(1, jump_max)
        else:
            jump = 1

        current_pos += jump

        if current_pos >= n_total:
            break

        remaining_missing = n_missing - len(missing_indices)
        upper_bound = min(max_segment, remaining_missing + 1)

        # Calculate missing segment length (avoid randint parameter error)
        if upper_bound <= min_segment:
            segment_length = min(remaining_missing, n_total - current_pos)
        else:
            segment_length = np.random.randint(min_segment, upper_bound)

        end_pos = min(current_pos + segment_length, n_total)
        missing_indices.extend(range(current_pos, end_pos))
        current_pos = end_pos

    # Ensure accurate missing count
    missing_indices = sorted(list(set(missing_indices)))[:n_missing]

    if len(missing_indices) < n_missing:
        available = [i for i in range(n_total) if i not in missing_indices]
        if available:
            additional = np.random.choice(
                available,
                size=min(n_missing - len(missing_indices), len(available)),
                replace=False
            )
            missing_indices.extend(additional)
            missing_indices = sorted(list(set(missing_indices)))[:n_missing]

    masked_data.iloc[missing_indices] = np.nan
    return masked_data, missing_indices


def calculate_metrics(true_values, predicted_values, missing_indices):
    """
    Calculate evaluation metrics

    Parameters:
        true_values: Series, true values (complete data)
        predicted_values: array-like, predicted values (only missing positions)
        missing_indices: DatetimeIndex or list, indices of missing positions
    """
    # Ensure missing_indices is a valid index
    if isinstance(missing_indices, pd.DatetimeIndex):
        valid_indices = missing_indices
    else:
        valid_indices = pd.DatetimeIndex(missing_indices)

    # Convert predicted values to Series using missing_indices as index
    if isinstance(predicted_values, pd.Series):
        pred_series = predicted_values
    else:
        pred_series = pd.Series(predicted_values, index=valid_indices)

    # Extract true and predicted values
    true_missing = true_values.loc[valid_indices]
    pred_missing = pred_series.loc[valid_indices]

    # Filter valid values
    valid_mask = true_missing.notna() & pred_missing.notna() & np.isfinite(pred_missing)

    if valid_mask.sum() == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'correlation': np.nan}

    true_clean = true_missing[valid_mask].values
    pred_clean = pred_missing[valid_mask].values

    # RMSE & MAE
    rmse = np.sqrt(mean_squared_error(true_clean, pred_clean))
    mae = mean_absolute_error(true_clean, pred_clean)

    # Correlation
    try:
        correlation, _ = pearsonr(true_clean, pred_clean)
        if np.isnan(correlation):
            correlation = 1.0 if np.array_equal(true_clean, pred_clean) else 0.0
    except:
        correlation = 0.0

    return {
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation
    }


def run_spatial_lgbm_experiments(target_data, neighbor_data,
                                 missing_days_list, missing_ratios,
                                 output_folder):
    """
    Run complete LightGBM + spatial correlation fusion interpolation experiments
    """
    print("\n" + "=" * 80)
    print("🚀 LightGBM + Spatial Correlation Fusion Interpolation Experiments")
    print("=" * 80)
    # ✅ Define global model configuration
    MODELS_CONFIG = {
        'baseline_time': {
            'name': 'Baseline: Time Only',
            'groups': ['time'],
            'description': 'Time features only'
        },
        'baseline_history': {
            'name': 'Baseline: Time + History',
            'groups': ['time', 'target_history'],
            'description': 'Time + target station history'
        },
        'ablation_basic_neighbor': {
            'name': 'Ablation: + Basic Neighbor',
            'groups': ['time', 'target_history', 'neighbor_basic'],
            'description': '+ Basic neighbor features (core spatial information)'
        },
        'ablation_with_meta': {
            'name': 'Ablation: + Spatial Meta',
            'groups': ['time', 'target_history', 'neighbor_basic', 'spatial_meta'],
            'description': '+ Static spatial meta features'
        },
        'ablation_with_dynamic': {
            'name': 'Ablation: + Dynamic Spatial',
            'groups': ['time', 'target_history', 'neighbor_basic',
                       'spatial_meta', 'spatial_dynamic'],
            'description': '+ Dynamic spatial features'
        },
        'full_model': {
            'name': 'Full Model',
            'groups': ['time', 'target_history', 'neighbor_basic',
                       'spatial_meta', 'spatial_dynamic', 'bias_correction'],
            'description': 'Full model (with Bias correction)'
        },
        'linear': {
            'name': 'Baseline: Linear Interpolation',
            'groups': None,
            'description': 'Linear interpolation baseline'
        }
    }
    # Initialize feature engineering
    feature_engineer = SpatialCorrelationFeatureEngineering(
        correlation_r=CORRELATION_R,
        neighbor_dist_km=NEIGHBOR_DIST_KM,
        bias_value=BIAS_VALUE
    )

    # Prepare data indices
    total_length = len(target_data)
    train_end = int(0.6 * total_length)
    val_end = int(0.8 * total_length)

    train_idx = target_data.index[:train_end]
    val_idx = target_data.index[train_end:val_end]
    test_idx = target_data.index[val_end:]

    print(f"\nData Split:")
    print(f"  Train: {len(train_idx)} days ({train_idx[0].date()} ~ {train_idx[-1].date()})")
    print(f"  Val:   {len(val_idx)} days ({val_idx[0].date()} ~ {val_idx[-1].date()})")
    print(f"  Test:  {len(test_idx)} days ({test_idx[0].date()} ~ {test_idx[-1].date()})")

    # Fit Bias correction on training set
    print("\nFitting Bias correction model...")
    feature_engineer.fit_bias_correction(
        target_data.loc[train_idx.union(val_idx)],
        neighbor_data.loc[train_idx.union(val_idx)]
    )

    results_list = []

    # ====================================================================
    # Experiment 1: Continuous missing data
    # ====================================================================
    print("\n" + "=" * 80)
    print("🔴 Experiment 1: Continuous Missing Segments")
    print("=" * 80)

    for missing_days in missing_days_list:
        print(f"\n{'─' * 80}")
        print(f"Missing segment length: {missing_days} days")
        print(f"{'─' * 80}")

        # Initialize result storage
        method_results = {key: {'rmse': [], 'mae': [], 'corr': []}
                          for key in MODELS_CONFIG.keys()}

        for repeat in range(N_REPEATS):
            print(f"\n  Repeat {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat

            # Create missing data (only in Test set)
            test_data = target_data.loc[test_idx].copy()
            n_segments = max(1, int(len(test_idx) * 0.1 / missing_days))

            masked_test, missing_indices_local, _ = create_continuous_missing(
                test_data, n_segments, missing_days, seed=seed
            )

            missing_indices_global = test_idx[missing_indices_local]

            # Build complete target data
            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # Build complete feature matrix
            df_features = feature_engineer.create_features(
                full_target, neighbor_data,
                n_lags=7, rolling_windows=[3, 7, 14],
                include_target_lags=True, use_neighbor_lag0=False
            )

            # Train and evaluate each model
            for model_key, model_cfg in MODELS_CONFIG.items():
                try:
                    if model_key == 'linear':
                        # Special handling: Linear interpolation
                        pred = BaselineInterpolators.linear_interpolation(
                            full_target, missing_indices_global
                        )
                    else:
                        # Select features
                        selected_features = feature_engineer.select_features_by_groups(
                            df_features, model_cfg['groups']
                        )

                        # Prepare training data
                        train_df = df_features.loc[train_idx.union(val_idx)].dropna(subset=['target'])
                        X_train = train_df[selected_features].fillna(train_df[selected_features].mean())
                        y_train = train_df['target']

                        # Train model
                        model = lgb.LGBMRegressor(**LGBM_PARAMS)
                        model.fit(X_train, y_train)

                        # Predict
                        X_test = df_features.loc[missing_indices_global][selected_features]
                        X_test = X_test.fillna(X_train.mean())
                        pred = model.predict(X_test)

                    # Calculate metrics
                    metrics = calculate_metrics(target_data, pred, missing_indices_global)

                    method_results[model_key]['rmse'].append(metrics['rmse'])
                    method_results[model_key]['mae'].append(metrics['mae'])
                    method_results[model_key]['corr'].append(metrics['correlation'])

                    print(f"    ✓ {model_cfg['name']}: RMSE={metrics['rmse']:.4f}")

                except Exception as e:
                    print(f"    ✗ {model_cfg['name']} failed: {e}")
                    method_results[model_key]['rmse'].append(np.nan)
                    method_results[model_key]['mae'].append(np.nan)
                    method_results[model_key]['corr'].append(np.nan)

        # Summarize results
        for model_key, model_cfg in MODELS_CONFIG.items():
            method_data = method_results[model_key]
            if len([x for x in method_data['rmse'] if not np.isnan(x)]) > 0:
                results_list.append({
                    'experiment': 'continuous',
                    'station': TARGET_STATION,
                    'missing_days': missing_days,
                    'method': model_key,
                    'method_description': model_cfg['description'],
                    'rmse_mean': np.nanmean(method_data['rmse']),
                    'rmse_std': np.nanstd(method_data['rmse']),
                    'mae_mean': np.nanmean(method_data['mae']),
                    'mae_std': np.nanstd(method_data['mae']),
                    'correlation_mean': np.nanmean(method_data['corr']),
                    'correlation_std': np.nanstd(method_data['corr']),
                    'n_repeats': N_REPEATS
                })

# ====================================================================
# Experiment 2: Random Missing Data
# ====================================================================
print("\n" + "=" * 80)
print("🔴 Experiment 2: Random Missing Data")
print("=" * 80)

for missing_ratio in missing_ratios:
    print(f"\n{'─' * 80}")
    print(f"Missing Ratio: {missing_ratio * 100:.1f}%")  # ✅ Fixed print content
    print(f"{'─' * 80}")

    # Initialize result storage
    # Initialize result storage
    method_results = {key: {'rmse': [], 'mae': [], 'corr': []}
                      for key in MODELS_CONFIG.keys()}  # ✅ Changed to MODELS_CONFIG

    for repeat in range(N_REPEATS):
        print(f"\n  Repeat {repeat + 1}/{N_REPEATS}")

        seed = RANDOM_SEED + repeat

        # Create missing data (only in Test set)
        test_data = target_data.loc[test_idx].copy()
        # n_segments = max(1, int(len(test_idx) * 0.1 / missing_ratio))

        # ✅ Fixed: Use create_random_clustered_missing
        masked_test, missing_indices_local = create_random_clustered_missing(
            test_data, missing_ratio, min_segment=5, max_segment=30, seed=seed
        )

        missing_indices_global = test_idx[missing_indices_local]

        # Build complete data
        full_target = target_data.copy()
        full_target.loc[missing_indices_global] = np.nan

        # ✅ Fixed: lag0 should be enabled for random missing data
        df_features = feature_engineer.create_features(
            full_target, neighbor_data,
            n_lags=7, rolling_windows=[3, 7, 14],
            include_target_lags=True, use_neighbor_lag0=True  # ✅ Changed to True
        )

        # Train and evaluate each model
        for model_key, model_cfg in MODELS_CONFIG.items():
            try:
                if model_key == 'linear':
                    # Special handling: Linear interpolation
                    pred = BaselineInterpolators.linear_interpolation(
                        full_target, missing_indices_global
                    )
                else:
                    # Select features
                    selected_features = feature_engineer.select_features_by_groups(
                        df_features, model_cfg['groups']
                    )

                    # Prepare training data
                    train_df = df_features.loc[train_idx.union(val_idx)].dropna(subset=['target'])
                    X_train = train_df[selected_features].fillna(train_df[selected_features].mean())
                    y_train = train_df['target']

                    # Train model
                    model = lgb.LGBMRegressor(**LGBM_PARAMS)
                    model.fit(X_train, y_train)

                    # Predict
                    X_test = df_features.loc[missing_indices_global][selected_features]
                    X_test = X_test.fillna(X_train.mean())
                    pred = model.predict(X_test)

                # Calculate metrics
                metrics = calculate_metrics(target_data, pred, missing_indices_global)

                method_results[model_key]['rmse'].append(metrics['rmse'])
                method_results[model_key]['mae'].append(metrics['mae'])
                method_results[model_key]['corr'].append(metrics['correlation'])

                print(f"    ✓ {model_cfg['name']}: RMSE={metrics['rmse']:.4f}")

            except Exception as e:
                print(f"    ✗ {model_cfg['name']} failed: {e}")
                method_results[model_key]['rmse'].append(np.nan)
                method_results[model_key]['mae'].append(np.nan)
                method_results[model_key]['corr'].append(np.nan)

# Summarize results
for model_key, model_cfg in MODELS_CONFIG.items():
    method_data = method_results[model_key]
    if len([x for x in method_data['rmse'] if not np.isnan(x)]) > 0:
        results_list.append({
            'experiment': 'random',
            'station': TARGET_STATION,
            'missing_ratio': missing_ratio * 100,  # ✅ Convert to percentage
            'method': model_key,
            'method_description': model_cfg['description'],
            'rmse_mean': np.nanmean(method_data['rmse']),
            'rmse_std': np.nanstd(method_data['rmse']),
            'mae_mean': np.nanmean(method_data['mae']),
            'mae_std': np.nanstd(method_data['mae']),
            'correlation_mean': np.nanmean(method_data['corr']),
            'correlation_std': np.nanstd(method_data['corr']),
            'n_repeats': N_REPEATS
        })

# Save results
results_df = pd.DataFrame(results_list)
results_path = os.path.join(output_folder, 'lgbm_spatial_results.csv')
results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
print(f"\n✓ Results saved to: {results_path}")
# ========== New: Feature Group Contribution Analysis ==========
print("\n" + "=" * 80)
print("📊 Feature Group Incremental Contribution Analysis")
print("=" * 80)

def analyze_incremental_contribution(results_df, experiment_type='continuous'):
    """Analyze incremental contribution of each feature group"""
    exp_data = results_df[results_df['experiment'] == experiment_type]

    # Calculate average RMSE grouped by model
    model_order = [
        'baseline_time',
        'baseline_history',
        'ablation_basic_neighbor',
        'ablation_with_meta',
        'ablation_with_dynamic',
        'full_model'
    ]

    avg_rmse = {}
    for model in model_order:
        model_data = exp_data[exp_data['method'] == model]
        if len(model_data) > 0:
            avg_rmse[model] = model_data['rmse_mean'].mean()

    # Calculate incremental improvement
    print(f"\nFeature Group Incremental Contribution for {experiment_type.upper()} Experiment:\n")
    print(f"{'Stage':<30} {'Avg RMSE':<12} {'Rel Improvement':<12} {'Abs Improvement':<12}")
    print("─" * 70)

    baseline_rmse = avg_rmse.get('baseline_time', np.nan)
    prev_rmse = baseline_rmse

    for i, model in enumerate(model_order):
        if model in avg_rmse:
            current_rmse = avg_rmse[model]

            # Improvement compared to previous stage
            if i == 0:
                rel_improve = 0.0
                abs_improve = 0.0
            else:
                rel_improve = (prev_rmse - current_rmse) / prev_rmse * 100
                abs_improve = prev_rmse - current_rmse

            model_name = MODELS_CONFIG[model]['description']  # ✅ Changed to MODELS_CONFIG
            print(f"{model_name:<30} {current_rmse:>10.4f}  {rel_improve:>10.2f}%  {abs_improve:>10.4f}")

            prev_rmse = current_rmse

    # Key contribution percentages
    if 'baseline_history' in avg_rmse and 'ablation_basic_neighbor' in avg_rmse:
        spatial_contrib = (avg_rmse['baseline_history'] - avg_rmse['ablation_basic_neighbor']) / \
                          (avg_rmse['baseline_time'] - avg_rmse['full_model']) * 100
        print(f"\n🎯 Contribution of core spatial information (basic neighbor features): {spatial_contrib:.1f}% of total improvement")

    if 'ablation_basic_neighbor' in avg_rmse and 'full_model' in avg_rmse:
        advanced_contrib = (avg_rmse['ablation_basic_neighbor'] - avg_rmse['full_model']) / \
                           (avg_rmse['baseline_time'] - avg_rmse['full_model']) * 100
        print(f"🎯 Contribution of advanced spatial features (meta + dynamic + Bias): {advanced_contrib:.1f}% of total improvement")

# Analyze both experiments
analyze_incremental_contribution(results_df, 'continuous')
analyze_incremental_contribution(results_df, 'random')

return results_df, None  # Changed second return value to None (no single final_interpolator)


def plot_feature_importance(interpolator, output_folder):
    """Plot feature importance"""
    importance_df = interpolator.get_feature_importance(top_n=20)

    if importance_df is None:
        return

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'].values)
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Feature Importance', fontsize=13)
    plt.title('LightGBM Feature Importance (Top 20)', fontsize=15, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to: {save_path}")
    plt.close()


def plot_comparison_results(results_df, output_folder):
    """Plot method comparison results"""
    # Define model display order and colors
    model_display_order = [
        ('baseline_time', 'Time Only', 'lightcoral'),
        ('baseline_history', '+ History', 'orange'),
        ('ablation_basic_neighbor', '+ Neighbor', 'gold'),
        ('ablation_with_meta', '+ Spatial Meta', 'lightgreen'),
        ('ablation_with_dynamic', '+ Dynamic', 'skyblue'),
        ('full_model', 'Full Model', 'steelblue'),
        ('linear', 'Linear Interp.', 'gray')
    ]

    # Figure 1: Continuous Missing - RMSE Comparison
    continuous_df = results_df[results_df['experiment'] == 'continuous']

    if len(continuous_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE
        pivot_rmse = continuous_df.pivot_table(
            values='rmse_mean', index='missing_days', columns='method'
        )

        # ✅ Correct
        ax = axes[0]
        for model_key, display_name, color in model_display_order:
            if model_key in pivot_rmse.columns:
                model_data = pivot_rmse[model_key]
                ax.plot(pivot_rmse.index, model_data,
                        marker='o', label=display_name,
                        linewidth=2.5, color=color, alpha=0.8)

        ax.set_xlabel('Missing Days', fontsize=12)
        ax.set_ylabel('RMSE(mm)', fontsize=12)
        ax.set_title('Continuous Missing(RMSE)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MAE
        pivot_mae = continuous_df.pivot_table(
            values='mae_mean', index='missing_days', columns='method'
        )

        ax = axes[1]
        for model_key, display_name, color in model_display_order:
            if model_key in pivot_mae.columns:
                model_data = pivot_mae[model_key]
                ax.plot(pivot_mae.index, model_data,
                        marker='s', label=display_name,
                        linewidth=2.5, color=color, alpha=0.8)

        ax.set_xlabel('Missing Days', fontsize=12)
        ax.set_ylabel('MAE(mm)', fontsize=12)
        ax.set_title('Continuous Missing(MAE)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_folder, 'continuous_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Continuous missing comparison plot saved to: {save_path}")
        plt.close()

    # Figure 2: Random Missing - RMSE Comparison
    random_df = results_df[results_df['experiment'] == 'random']

    if len(random_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pivot_rmse = random_df.pivot_table(
            values='rmse_mean', index='missing_ratio', columns='method'
        )

        ax = axes[0]
        for model_key, display_name, color in model_display_order:
            if model_key in pivot_rmse.columns:
                model_data = pivot_rmse[model_key]
                ax.plot(pivot_rmse.index, model_data,
                        marker='o', label=display_name,
                        linewidth=2.5, color=color, alpha=0.8)

        ax.set_xlabel('Missing Ratio(%)', fontsize=12)
        ax.set_ylabel('RMSE(mm)', fontsize=12)
        ax.set_title('Random Missing(RMSE)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        pivot_mae = random_df.pivot_table(
            values='mae_mean', index='missing_ratio', columns='method'
        )

        ax = axes[1]
        for model_key, display_name, color in model_display_order:
            if model_key in pivot_mae.columns:
                model_data = pivot_mae[model_key]
                ax.plot(pivot_mae.index, model_data,
                        marker='s', label=display_name,
                        linewidth=2.5, color=color, alpha=0.8)

        ax.set_xlabel('Missing Ratio(%)', fontsize=12)
        ax.set_ylabel('MAE(mm)', fontsize=12)
        ax.set_title('Random Missing(MAE)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_folder, 'random_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Random missing comparison plot saved to: {save_path}")
        plt.close()


# =====================================================================
# SHAP Visualization Functions (added after plot_comparison_results)
# =====================================================================

def plot_shap_summary(interpolator, output_folder, top_n=20):
    """
    Plot SHAP summary plots

    Parameters:
        interpolator: Trained interpolator (contains SHAP values)
        output_folder: Output folder path
        top_n: Show top N features
    """
    if interpolator.shap_values is None or interpolator.X_sample_for_shap is None:
        print("    [Warning] SHAP values not found, skipping plotting")
        return

    X_sample = interpolator.X_sample_for_shap
    shap_values = interpolator.shap_values

    # Figure 1: Feature importance bar chart
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar",
                          max_display=top_n, show=False)
        plt.title('SHAP Feature Importance (Global)', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_folder, 'shap_importance_bar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ SHAP bar plot saved to: {save_path}")
        plt.close()
    except Exception as e:
        print(f"    [Warning] Failed to plot SHAP bar chart: {e}")
        plt.close()

    # Figure 2: Beeswarm plot (show feature distribution and impact direction)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, max_display=top_n, show=False)
        plt.title('SHAP Feature Distribution', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_folder, 'shap_summary_beeswarm.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ SHAP beeswarm plot saved to: {save_path}")
        plt.close()
    except Exception as e:
        print(f"    [Warning] Failed to plot SHAP beeswarm plot: {e}")
        plt.close()

def plot_shap_dependence(interpolator, feature_name, output_folder):
    """
    Plot SHAP dependence plot (detailed analysis of a single feature)

    Parameters:
        interpolator: Trained interpolator
        feature_name: Name of the feature to analyze
        output_folder: Output folder path
    """
    if interpolator.shap_values is None or interpolator.X_sample_for_shap is None:
        return

    X_sample = interpolator.X_sample_for_shap
    shap_values = interpolator.shap_values

    try:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name, shap_values, X_sample,
            interaction_index="auto",
            show=False
        )
        plt.title(f'SHAP Dependence: {feature_name}', fontsize=15, fontweight='bold')
        plt.tight_layout()

        safe_filename = feature_name.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(output_folder, f'shap_dependence_{safe_filename}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ SHAP dependence plot saved: {save_path}")
        plt.close()
    except Exception as e:
        print(f"    [Warning] Failed to plot SHAP dependence ({feature_name}): {e}")
        plt.close()


def plot_feature_importance_comparison(interpolator, output_folder):
    """
    Compare feature importance between LightGBM and SHAP

    Parameters:
        interpolator: Trained interpolator (SHAP must be computed first)
        output_folder: Output folder path
    """
    if interpolator.shap_values is None:
        print("    [Warning] SHAP values not computed, skipping comparison plot")
        return None

    try:
        comparison_df = interpolator.compare_feature_importance_methods()

        top_features = comparison_df.head(15)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        ax = axes[0]
        x = np.arange(len(top_features))
        width = 0.35

        ax.barh(x - width / 2, top_features['lgbm_norm'], width,
                label='LightGBM', alpha=0.8, color='steelblue')
        ax.barh(x + width / 2, top_features['shap_norm'], width,
                label='SHAP', alpha=0.8, color='coral')

        ax.set_yticks(x)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Normalized Importance', fontsize=12)
        ax.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        ax = axes[1]
        colors = ['green' if d <= 3 else 'orange' if d <= 5 else 'red'
                  for d in top_features['rank_diff']]

        ax.barh(range(len(top_features)), top_features['rank_diff'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Rank Difference', fontsize=12)
        ax.set_title('Ranking Inconsistency', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_folder, 'importance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Feature importance comparison plot saved: {save_path}")
        plt.close()

        return comparison_df

    except Exception as e:
        print(f"    [Warning] Failed to plot feature importance comparison: {e}")
        plt.close()
        return None


def analyze_spatial_feature_contribution_with_shap(interpolator, output_folder):
    """
    Analyze spatial feature contribution using SHAP

    Parameters:
        interpolator: Trained interpolator (SHAP must be computed first)
        output_folder: Output folder path

    Returns:
        spatial_contribution_pct: Percentage of spatial feature contribution
    """
    if interpolator.shap_values is None:
        print("    [Warning] SHAP values not computed, skipping spatial feature analysis")
        return 0.0

    shap_values = interpolator.shap_values

    spatial_keywords = ['neighbor_r', 'neighbor_dist', 'spatial_weight',
                        'nbr_adj', 'nbr_lag', 'nbr_roll', 'nbr_isnan',
                        'spatial_corr', 'nbr_tgt_diff', 'nbr_consecutive']

    spatial_features = [f for f in interpolator.feature_names if any(
        keyword in f for keyword in spatial_keywords
    )]

    if len(spatial_features) == 0:
        print("    [Warning] No spatial features found")
        return 0.0

    spatial_indices = [interpolator.feature_names.index(f) for f in spatial_features]
    spatial_shap = np.abs(shap_values[:, spatial_indices]).sum(axis=1).mean()

    total_shap = np.abs(shap_values).sum(axis=1).mean()
    spatial_contribution_pct = (spatial_shap / total_shap) * 100

    print("\n" + "=" * 60)
    print("🎯 SHAP Spatial Feature Contribution Analysis")
    print("=" * 60)
    print(f"Total spatial features: {len(spatial_features)}")
    print(f"Spatial feature contribution: {spatial_contribution_pct:.2f}%")
    print(f"Other feature contribution: {100 - spatial_contribution_pct:.2f}%")
    print("\nTop 5 spatial features:")

    spatial_importance = pd.DataFrame({
        'feature': spatial_features,
        'shap_importance': [np.abs(shap_values[:, interpolator.feature_names.index(f)]).mean()
                            for f in spatial_features]
    }).sort_values('shap_importance', ascending=False)

    print(spatial_importance.head(5).to_string(index=False))

    save_path = os.path.join(output_folder, 'spatial_features_shap_analysis.csv')
    spatial_importance.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Detailed results saved: {save_path}")

    return spatial_contribution_pct


def run_shap_analysis_pipeline(interpolator, df_features, train_idx, output_folder):
    """
    Complete SHAP analysis pipeline

    Parameters:
        interpolator: Trained interpolator
        df_features: Complete feature DataFrame
        train_idx: Training set indices
        output_folder: Output folder

    Returns:
        shap_results: Dictionary containing all SHAP analysis results
    """
    print("\n" + "=" * 80)
    print("🔍 Starting SHAP Feature Contribution Analysis")
    print("=" * 80)

    X_train = df_features.loc[train_idx].drop(columns=['target', 'neighbor'], errors='ignore')
    X_train = X_train.fillna(X_train.mean())

    print("\nComputing SHAP values...")
    shap_values, X_sample = interpolator.compute_shap_values(X_train, max_samples=1000)

    if shap_values is None:
        print("    [Error] SHAP computation failed, skipping subsequent analysis")
        return None

    shap_results = {}

    print("\nGenerating SHAP visualization plots...")
    plot_shap_summary(interpolator, output_folder, top_n=20)

    comparison_df = plot_feature_importance_comparison(interpolator, output_folder)
    if comparison_df is not None:
        save_path = os.path.join(output_folder, 'importance_comparison.csv')
        comparison_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"    ✓ Comparison results saved: {save_path}")
        shap_results['comparison'] = comparison_df

    spatial_contrib = analyze_spatial_feature_contribution_with_shap(
        interpolator, output_folder
    )
    shap_results['spatial_contribution_pct'] = spatial_contrib

    print("\nGenerating key feature dependence plots...")
    shap_importance_df = interpolator.get_shap_feature_importance(top_n=5)
    for i, feature in enumerate(shap_importance_df['feature'].head(3)):
        print(f"  [{i + 1}/3] Plotting feature: {feature}")
        plot_shap_dependence(interpolator, feature, output_folder)

    shap_results['top_features'] = shap_importance_df

    print("\n" + "=" * 80)
    print("✅ SHAP analysis completed!")
    print("=" * 80)
    print(f"Total spatial feature contribution: {spatial_contrib:.2f}%")

    return shap_results


def generate_summary_report(results_df, output_folder):
    """Generate summary report"""
    print("\n" + "=" * 80)
    print("📊 Generating summary report")
    print("=" * 80)

    summary_overall = results_df.groupby('method').agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean',
    }).round(3)

    print("\nOverall average performance:")
    print(summary_overall)

    overall_path = os.path.join(output_folder, 'summary_overall.csv')
    summary_overall.to_csv(overall_path, encoding='utf-8-sig')

    summary_by_exp = results_df.groupby(['experiment', 'method']).agg({
        'rmse_mean': ['mean', 'std'],
        'mae_mean': ['mean', 'std'],
        'correlation_mean': ['mean', 'std']
    }).round(4)

    print("\nSummary by experiment type:")
    print(summary_by_exp)

    exp_path = os.path.join(output_folder, 'summary_by_experiment.csv')
    summary_by_exp.to_csv(exp_path, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print("🏆 Method ranking (based on average RMSE)")
    print("=" * 80)

    ranking = results_df.groupby('method')['rmse_mean'].mean().sort_values()
    for rank, (method, rmse) in enumerate(ranking.items(), 1):
        print(f"  {rank}. {method:20s} - RMSE: {rmse:.4f} mm")

    if 'lgbm_full' in ranking.index and 'linear' in ranking.index:
        improvement = (ranking['linear'] - ranking['lgbm_full']) / ranking['linear'] * 100
        print(f"\n✨ LGBM_Full improvement over Linear: {improvement:.2f}%")

    if 'lgbm_full' in ranking.index and 'lgbm_no_spatial' in ranking.index:
        spatial_benefit = (ranking['lgbm_no_spatial'] - ranking['lgbm_full']) / ranking['lgbm_no_spatial'] * 100
        print(f"✨ Spatial feature contribution: {spatial_benefit:.2f}%")


# =====================================================================
# Main Program
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LightGBM + Spatial Correlation Fusion Interpolation System")
    print("=" * 80)
    print(f"Configuration Info:")
    print(f"  - Target Station: {TARGET_STATION}")
    print(f"  - Neighbor Station: {NEIGHBOR_STATION}")
    print(f"  - Spatial Correlation: r = {CORRELATION_R:.4f}")
    print(f"  - Station Distance: {NEIGHBOR_DIST_KM:.2f} km")
    print(f"  - Analysis Period: {START_DATE} to {END_DATE}")
    print(f"  - Experiment Repeats: {N_REPEATS}")
    print("=" * 80)

    print("\nLoading data...")
    try:
        df_target = pd.read_csv(
            os.path.join(FOLDER, f"{TARGET_STATION}.csv"),
            parse_dates=[TIME_COL]
        )
        df_neighbor = pd.read_csv(
            os.path.join(FOLDER, f"{NEIGHBOR_STATION}.csv"),
            parse_dates=[TIME_COL]
        )

        df_target = df_target.set_index(TIME_COL).sort_index()
        df_neighbor = df_neighbor.set_index(TIME_COL).sort_index()

        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)

        target_data = df_target.loc[start_dt:end_dt, VALUE_COL]
        neighbor_data = df_neighbor.loc[start_dt:end_dt, VALUE_COL]

        full_date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        target_data = target_data.reindex(full_date_range)
        neighbor_data = neighbor_data.reindex(full_date_range)

        print(f"✓ Data loaded successfully")
        print(f"  - Target station: {len(target_data)} days, missing {target_data.isna().sum()} days")
        print(f"  - Neighbor station: {len(neighbor_data)} days, missing {neighbor_data.isna().sum()} days")

    except Exception as e:
        print(f"[Error] Data loading failed: {e}")
        exit(1)

    output_folder = os.path.join(FOLDER, "lgbm_spatial_results")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder: {output_folder}")

    sample_length = len(target_data)

    if sample_length > 1000:
        missing_days_list = [7, 15, 30, 60, 90, 120, 180]
    elif sample_length > 500:
        missing_days_list = [7, 15, 30, 60]
    else:
        missing_days_list = [7, 15, 30]

    missing_ratios = np.arange(0.05, 0.55, 0.05)

    print(f"\nExperiment Configuration:")
    print(f"  - Continuous missing days: {missing_days_list}")
    print(f"  - Random missing ratios: {[f'{r * 100:.0f}%' for r in missing_ratios]}")

    results_df,_ = run_spatial_lgbm_experiments(
        target_data, neighbor_data,
        missing_days_list, missing_ratios,
        output_folder
    )

    print("\n" + "=" * 80)
    print("Generating visualization plots...")
    print("=" * 80)

    plot_comparison_results(results_df, output_folder)

    generate_summary_report(results_df, output_folder)

    config_info = {
        'target_station': TARGET_STATION,
        'neighbor_station': NEIGHBOR_STATION,
        'correlation_r': CORRELATION_R,
        'neighbor_dist_km': NEIGHBOR_DIST_KM,
        'bias_value': BIAS_VALUE,
        'time_range': f"{START_DATE} to {END_DATE}",
        'data_length': sample_length,
        'n_repeats': N_REPEATS,
        'lgbm_params': LGBM_PARAMS,
        'missing_days_list': missing_days_list,
        'missing_ratios': missing_ratios.tolist(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print("\n" + "=" * 80)
    print("🔬 Performing SHAP feature contribution analysis...")
    print("=" * 80)

    try:
        total_length = len(target_data)
        train_end = int(0.6 * total_length)
        val_end = int(0.8 * total_length)

        train_idx = target_data.index[:train_end]
        val_idx = target_data.index[train_end:val_end]
        train_val_idx = target_data.index[:val_end]

        feature_engineer_for_shap = SpatialCorrelationFeatureEngineering(
            correlation_r=CORRELATION_R,
            neighbor_dist_km=NEIGHBOR_DIST_KM,
            bias_value=BIAS_VALUE
        )

        train_val_idx = target_data.index[:val_end]
        feature_engineer_for_shap.fit_bias_correction(
            target_data.loc[train_val_idx],
            neighbor_data.loc[train_val_idx]
        )

        df_features_for_shap = feature_engineer_for_shap.create_features(
            target_data, neighbor_data,
            n_lags=7, rolling_windows=[3, 7, 14],
            include_target_lags=True, use_neighbor_lag0=True
        )

        print("    Training full model for SHAP analysis...")
        shap_interpolator = LightGBMSpatialInterpolator(feature_engineer_for_shap, LGBM_PARAMS)
        shap_interpolator.train(df_features_for_shap, train_val_idx)

        shap_results = run_shap_analysis_pipeline(
            shap_interpolator,
            df_features_for_shap,
            train_val_idx,
            output_folder
        )

        if shap_results is not None:
            config_info['shap_analysis'] = {
                'spatial_contribution_pct': shap_results.get('spatial_contribution_pct', 0.0),
                'top_5_features': shap_results['top_features'].head(5)[
                    'feature'].tolist() if 'top_features' in shap_results else []
            }

    except Exception as e:
        print(f"\n[Warning] SHAP analysis failed: {e}")
        print("Continuing with subsequent processes...")

    config_path = os.path.join(output_folder, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False)
