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
import optuna  
from optuna.samplers import TPESampler  
import json

warnings.filterwarnings("ignore")

# ----------------------
# Uniform font settings
# ----------------------
rcParams['font.family'] = ['SimHei', 'Times New Roman']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 15
rcParams['legend.fontsize'] = 11
rcParams['figure.titlesize'] = 16

# =====================================================================
# Key Configuration Parameters
# =====================================================================
FOLDER = "../data/real_GNSS"
TARGET_STATION = "YNYS"
NEIGHBOR_STATION = "YNLJ"
TIME_COL = "YYYYMMDD"
VALUE_COL = "U(m)"


# LightGBM Hyperparameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'num_leaves': 15,
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
# Hyperparameter Optimization Module
# =====================================================================
import optuna
from optuna.samplers import TPESampler


def optimize_lgbm_hyperparameters(target_data, neighbor_data, output_folder, n_trials=100):
    """
    Using Optuna for LightGBM Hyperparameter Optimization
    Ensure full consistency with the feature engineering process in the experimental workflow
    """
    print("\n" + "=" * 80)
    print("🔍 Optuna Bayesian Optimization ")
    print("=" * 80)


    # Prepare the data index (consistent with the experiment)
    total_length = len(target_data)
    train_end = int(0.6 * total_length)
    val_end = int(0.8 * total_length)

    train_idx = target_data.index[:train_end]
    val_idx = target_data.index[train_end:val_end]

    # ⚠️ Key: Use `train_idx.union(val_idx)` to fit the bias correction (consistent with the experiment)
    print("\nFitting the bias correction model...")
    feature_engineer.fit_bias_correction(
        target_data.loc[train_idx.union(val_idx)],
        neighbor_data.loc[train_idx.union(val_idx)]
    )

    # ⚠️ Key: Use the complete target_data (with no manually imputed missing values) when building features.
    df_features = feature_engineer.create_features(
        target_data,  
        neighbor_data,
        n_lags=7,
        rolling_windows=[3, 7, 14],
        include_target_lags=True,
        use_neighbor_lag0=True
    )

    # Prepare training/validation data (consistent with the experiment)
    train_df = df_features.loc[train_idx.union(val_idx)].dropna(subset=['target'])

    # ⚠️ Key: Split the data into a 8:2 training/validation split
    split_point = int(0.8 * len(train_df))

    train_split = train_df.iloc[:split_point]
    val_split = train_df.iloc[split_point:]

    # Feature Extraction and Object Detection
    X_train = train_split.drop(columns=['target', 'neighbor'], errors='ignore')
    y_train = train_split['target']
    X_train = X_train.fillna(X_train.mean())

    X_val = val_split.drop(columns=['target', 'neighbor'], errors='ignore')
    y_val = val_split['target']
    X_val = X_val.fillna(X_train.mean())  # Pad with the mean of the training set

    print(f"\nData Segmentation:")
    print(f"  training set: {len(X_train)} Sample")
    print(f"  Validation set: {len(X_val)} Sample")
    print(f"  Number of features: {X_train.shape[1]}")

    def objective(trial):
        """Optuna Objective function"""
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'random_state': RANDOM_SEED,

            # Hyperparameter search space
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=500),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)

        return mae

    # Performance Optimization
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED)
    )

    print(f"\nStart optimization ({n_trials} trials in total)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Output
    print("\n" + "=" * 80)
    print("✅ Optimization complete!")
    print("=" * 80)
  
    for key, value in study.best_params.items():
        print(f"  {key:20s}: {value}")


    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0])
        axes[0].set_title('Optimization History', fontsize=14, fontweight='bold')

        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[1])
        axes[1].set_title('Hyperparameter Importances', fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_folder, 'hyperparameter_optimization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")

    return study.best_params


# =====================================================================

class SpatialCorrelationFeatureEngineering:
    """Spatial Correlation Feature Engineering"""

    def __init__(self, correlation_r, neighbor_dist_km, bias_value):



    def fit_bias_correction(self, target_series, neighbor_series):
        """
        Fit an OLS bias-corrected model on the training set
        U_target = a + b * U_neighbor
        """
        valid_mask = target_series.notna() & neighbor_series.notna()

        if valid_mask.sum() < 30:
            print("    [Warning] Insufficient training data; using default parameters")
            self.ols_model = {'intercept': 0.0, 'coef': 1.0}
            return

        self.ols_model = {
            'intercept': lr.intercept_,
            'coef': lr.coef_[0]
        }

        print(f"    ✓ Bias校正: U_target = {lr.intercept_:.4f} + {lr.coef_[0]:.4f} * U_neighbor")

    def apply_bias_correction(self, neighbor_series):
        """Apply Bias Correction"""
        if self.ols_model is None:
            return neighbor_series.copy()

        return self.ols_model['intercept'] + self.ols_model['coef'] * neighbor_series

    def create_features(self, target_series, neighbor_series,
                        n_lags=7, rolling_windows=[3, 7, 14],
                        include_target_lags=True, use_neighbor_lag0=True):
        """
        Constructing a Complete Feature Matrix

        Feature Categories:
        1. Temporal Features: doy_sin, doy_cos, month, year
        2. Neighbor Station Lags: nbr_lag0–lagN
        3. Rolling statistics from neighboring stations: rollmean, rollstd (multi-window)
        4. Target station history: tgt_lag1–lag3 (past values only)
        5. Spatial meta-features: neighbor_r, neighbor_dist_km
        6. Bias correction features: nbr_adj_lag0
        7. Missing value indicators: nbr_isnan_lag0
        """
        # Build DataFrame
        df = pd.DataFrame({
            'target': target_series,
            'neighbor': neighbor_series
        })

        # === 1. Temporal characteristics ===
        df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)


        # === 2. 邻站滞后特征 (lag0允许用于离线插补) ===
        for lag in range(n_lags + 1):
            if lag == 0 and not use_neighbor_lag0:
                continue
            df[f'nbr_lag{lag}'] = df['neighbor'].shift(lag)

        # === 3. Neighboring station lag feature (lag0 is permitted for offline interpolation) ===
        for window in rolling_windows:
            # Note: Use `shift(1)` to ensure the current value is not included.
            df[f'nbr_rollmean_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).mean()
            df[f'nbr_rollstd_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).std().fillna(0)


        # === 4. Historical lag for the target station (past values only) ===
        if include_target_lags:
            for lag in [1, 2, 3, 7]:
                df[f'tgt_lag{lag}'] = df['target'].shift(lag)


        # === 5. Bias Calibration characteristics ===
        if self.ols_model is not None:
            nbr_adj = self.apply_bias_correction(df['neighbor'])
            for lag in range(min(3, n_lags + 1)):
                df[f'nbr_adj_lag{lag}'] = nbr_adj.shift(lag)

        # === 6. Missing metric ===
        df['nbr_isnan_lag0'] = df['neighbor'].isna().astype(int)
        df['n_neighbors_available'] = (~df['neighbor'].isna()).astype(int)

        return df


class LightGBMSpatialInterpolator:
    """A Spatially Correlated Interpolator Based on LightGBM"""

    def __init__(self, feature_engineer, lgbm_params=None):
        self.feature_engineer = feature_engineer
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.models = []  # Store the Bootstrap model collection
        self.feature_names = None

    def _prepare_train_data(self, df_features, train_idx):
        """Prepare training data"""
        train_df = df_features.loc[train_idx]

        # Keep only rows where target is not NaN
        train_df = train_df.dropna(subset=['target'])

        if len(train_df) < 50:
            raise ValueError(f"Insufficient training data: {len(train_df)}")

        # NaN values in features (replaced with the mean of the training set)
        X = X.fillna(X.mean())

        self.feature_names = X.columns.tolist()

        return X, y

    def train_single_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train a single LightGBM model"""
        model = lgb.LGBMRegressor(**self.lgbm_params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        else:
            model.fit(X_train, y_train)

        return model

    def get_feature_importance(self, top_n=15):
        """Obtain feature importance (averaged across all models)"""
        if not self.models:
            return None

        # Average feature importance across all models
        importances = np.zeros(len(self.feature_names))

        for model in self.models:
            importances += model.feature_importances_

        importances /= len(self.models)




class BaselineInterpolators:
    """Baseline interpolation methods"""

    @staticmethod
    def time_only_baseline(df_features, train_idx, test_idx):
        """
        Baseline A: Use time-based features only
        """
        time_features = ['doy_sin', 'doy_cos', 'month_norm', 'year_norm']

        train_df = df_features.loc[train_idx].dropna(subset=['target'])
        X_train = train_df[time_features].fillna(0)
        y_train = train_df['target']

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        test_df = df_features.loc[test_idx]
        X_test = test_df[time_features].fillna(0)
        pred = model.predict(X_test)

        return pred

    @staticmethod
    def time_plus_target_lags(df_features, train_idx, test_idx):
        """
        Baseline B: Time-based features + History of destination stations
        """
        features = ['doy_sin', 'doy_cos', 'month_norm', 'year_norm',
                    'tgt_lag1', 'tgt_lag2', 'tgt_lag3', 'tgt_lag7']

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
        """Linear Interpolation Baseline"""
        filled = series.copy()

        
        if isinstance(missing_indices, pd.DatetimeIndex):
            # Index directly using DatetimeIndex
            filled.loc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.loc[missing_indices].values
        else:
            # The case of integer indexes
            if not isinstance(missing_indices, (list, np.ndarray, pd.Index)):
                raise ValueError("missing_indices must be a list, numpy array or pandas Index")

            missing_indices = np.array(missing_indices, dtype=int)
            if len(missing_indices) == 0:
                return np.array([])

            # Ensure that the index is within the valid range
            if (missing_indices < 0).any() or (missing_indices >= len(series)).any():
                raise ValueError("Some indices are out of bounds")

            filled.iloc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.iloc[missing_indices].values

def create_continuous_missing(data, n_segments, days_per_segment, seed=None):
    """Create a continuous missing segment """
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


    masked_data.iloc[missing_indices] = np.nan
    return masked_data, sorted(list(set(missing_indices))), segments_created


def create_random_clustered_missing(data, missing_ratio, min_segment=5, max_segment=30, seed=None):
    """Create a random cluster with missing values"""
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

        # Calculate the jump distance
        if remaining > 30:
            jump = np.random.randint(5, min(30, remaining // 2))
        elif remaining > 2:
            jump_max = max(2, remaining // 2)  
            jump = np.random.randint(1, jump_max)
        else:
            jump = 1

        current_pos += jump

        if current_pos >= n_total:
            break

        remaining_missing = n_missing - len(missing_indices)
        upper_bound = min(max_segment, remaining_missing + 1)



    # Ensure the accuracy of the missing quantities
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


def calculate_metrics(true_values, predicted_values, pred_lower, pred_upper, missing_indices):
    """
    Calculate evaluation metrics + CI coverage

    Parameters:
        true_values: Series, actual values (complete data)
        predicted_values: array-like, predicted values (missing positions only)
        pred_lower: array-like, lower confidence bound
        pred_upper: array-like, upper confidence bound
        missing_indices: DatetimeIndex or list, indices of missing positions
    """
    # Ensure that `missing_indices` is a valid index
    if isinstance(missing_indices, pd.DatetimeIndex):
        valid_indices = missing_indices
    else:
        valid_indices = pd.DatetimeIndex(missing_indices)

    # Convert the predicted values to a Series, using `missing_indices` as the index
    if isinstance(predicted_values, pd.Series):
        pred_series = predicted_values
    else:
        pred_series = pd.Series(predicted_values, index=valid_indices)

    if pred_lower is not None and not isinstance(pred_lower, pd.Series):
        pred_lower = pd.Series(pred_lower, index=valid_indices)

    if pred_upper is not None and not isinstance(pred_upper, pd.Series):
        pred_upper = pd.Series(pred_upper, index=valid_indices)

    # Retrieve actual and forecast values
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
    Run the full LightGBM+spatial correlation experiment
    """
    print("\n" + "=" * 80)
    print("🚀 LightGBM + Experiments on Spatial Correlation Fusion Interpolation")
    print("=" * 80)

    # Initial Feature Engineering
    feature_engineer = SpatialCorrelationFeatureEngineering(
        correlation_r=CORRELATION_R,
        neighbor_dist_km=NEIGHBOR_DIST_KM,
        bias_value=BIAS_VALUE
    )

    # Prepare the data index
    total_length = len(target_data)
    train_end = int(0.6 * total_length)
    val_end = int(0.8 * total_length)

    train_idx = target_data.index[:train_end]
    val_idx = target_data.index[train_end:val_end]
    test_idx = target_data.index[val_end:]

    print(f"\nData Segmentation:")
    print(f"  Train: {len(train_idx)} Day ({train_idx[0].date()} ~ {train_idx[-1].date()})")
    print(f"  Val:   {len(val_idx)} Day ({val_idx[0].date()} ~ {val_idx[-1].date()})")
    print(f"  Test:  {len(test_idx)} Day ({test_idx[0].date()} ~ {test_idx[-1].date()})")

    # Fitting bias correction on the training set
    print("\nFitting the bias correction model...")
    feature_engineer.fit_bias_correction(
        target_data.loc[train_idx.union(val_idx)],
        neighbor_data.loc[train_idx.union(val_idx)]
    )

    results_list = []

    # ====================================================================
    # Experiment 1: Continuous Missing Values
    # ====================================================================
    print("\n" + "=" * 80)
    print("🔴 Experiment 1: Continuous Missing Segments")
    print("=" * 80)

    for missing_days in missing_days_list:
        print(f"\n{'─' * 80}")
        print(f"Length of missing segment: {missing_days} Day")
        print(f"{'─' * 80}")

        method_results = {
            'lgbm_full': {'rmse': [], 'mae': [], 'corr': []},
            'lgbm_no_spatial': {'rmse': [], 'mae': [], 'corr': []},
            'time_only': {'rmse': [], 'mae': [], 'corr': []},
            'linear': {'rmse': [], 'mae': [], 'corr': []}
        }

        for repeat in range(N_REPEATS):
            print(f"\n  repeat {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat
            # Convert to global index
            missing_indices_global = test_idx[missing_indices_local]

            # Build a complete dataset (Train and Val have no missing values; Test has missing values)
            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # === Feature construction ===
            df_features = feature_engineer.create_features(
                full_target, neighbor_data,
                n_lags=7, rolling_windows=[3, 7, 14],
                include_target_lags=True, use_neighbor_lag0=True
            )

            # === Model1: Full LightGBM  ===
            interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
            interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx))
            pred_median, pred_lower, pred_upper = interpolator.predict_with_uncertainty_residual(
                df_features, missing_indices_global
            )
            metrics_time = calculate_metrics(target_data, pred_time, missing_indices_global)

            method_results['lgbm_full']['rmse'].append(metrics['rmse'])
            method_results['lgbm_full']['mae'].append(metrics['mae'])
            method_results['lgbm_full']['corr'].append(metrics['correlation'])


            # === Model 2: LightGBM  ===
            # Remove spatial correlation features
            df_no_spatial = df_features.drop(columns=[
                                                         'neighbor_r', 'neighbor_dist_km', 'spatial_weight'
                                                     ] + [c for c in df_features.columns if 'nbr_adj' in c],
                                             errors='ignore')

            feature_engineer_nospatial = SpatialCorrelationFeatureEngineering(0, 0, 0)
            interpolator_nospatial = LightGBMSpatialInterpolator(feature_engineer_nospatial, LGBM_PARAMS)

            # Manually prepare training data
            train_df_ns = df_no_spatial.loc[train_idx.union(val_idx)].dropna(subset=['target'])
            X_train_ns = train_df_ns.drop(columns=['target', 'neighbor'], errors='ignore').fillna(train_df_ns.mean())
            y_train_ns = train_df_ns['target']

            model_ns = lgb.LGBMRegressor(**LGBM_PARAMS)
            model_ns.fit(X_train_ns, y_train_ns)

            X_test_ns = df_no_spatial.loc[missing_indices_global].drop(columns=['target', 'neighbor'],
                                                                       errors='ignore').fillna(X_train_ns.mean())
            pred_ns = model_ns.predict(X_test_ns)

            metrics_ns = calculate_metrics(
                target_data, pred_ns, None, None, missing_indices_global
            )

            method_results['lgbm_no_spatial']['rmse'].append(metrics_ns['rmse'])
            method_results['lgbm_no_spatial']['mae'].append(metrics_ns['mae'])
            method_results['lgbm_no_spatial']['corr'].append(metrics_ns['correlation'])

            # === Baseline: Time-only ===
            pred_time = BaselineInterpolators.time_only_baseline(
                df_features, train_idx.union(val_idx), missing_indices_global
            )

            metrics_time = calculate_metrics(
                target_data, pred_time, None, None, missing_indices_global
            )

            method_results['time_only']['rmse'].append(metrics_time['rmse'])
            method_results['time_only']['mae'].append(metrics_time['mae'])
            method_results['time_only']['corr'].append(metrics_time['correlation'])

            # === Baseline: Linear Interpolation ===
            pred_linear = BaselineInterpolators.linear_interpolation(
                target_data, missing_indices_global
            )

            metrics_linear = calculate_metrics_with_ci(
                target_data, pred_linear, None, None, missing_indices_global
            )

            method_results['linear']['rmse'].append(metrics_linear['rmse'])
            method_results['linear']['mae'].append(metrics_linear['mae'])
            method_results['linear']['corr'].append(metrics_linear['correlation'])

            print(f"    ✓ LGBM_NoSpatial: RMSE={metrics_ns['rmse']:.4f}")
            print(f"    ✓ Time_Only: RMSE={metrics_time['rmse']:.4f}")
            print(f"    ✓ Linear: RMSE={metrics_linear['rmse']:.4f}")

        # Summary of Results
        for method_name, method_data in method_results.items():
            if len(method_data['rmse']) > 0:
                results_list.append({
                    'experiment': 'continuous',
                    'station': TARGET_STATION,
                    'missing_days': missing_days,
                    'method': method_name,
                    'rmse_mean': np.nanmean(method_data['rmse']),
                    'rmse_std': np.nanstd(method_data['rmse']),
                    'mae_mean': np.nanmean(method_data['mae']),
                    'mae_std': np.nanstd(method_data['mae']),
                    'correlation_mean': np.nanmean(method_data['corr']),
                    'correlation_std': np.nanstd(method_data['corr']),
                    'n_repeats': N_REPEATS
                })

    # ====================================================================
    # Experiment 2: Random Clustering Missing
    # ====================================================================
    print("\n" + "=" * 80)
    print("🔵 Experiment 2: Random Clustering Missing")
    print("=" * 80)

    for missing_ratio in missing_ratios:
        print(f"\n{'─' * 80}")
        print(f"Missing proportion: {missing_ratio * 100:.1f}%")
        print(f"{'─' * 80}")

        method_results = {
            'lgbm_full': {'rmse': [], 'mae': [], 'corr': []},
            'lgbm_no_spatial': {'rmse': [], 'mae': [], 'corr': []},
            'time_only': {'rmse': [], 'mae': [], 'corr': []},
            'linear': {'rmse': [], 'mae': [], 'corr': []}
        }

        for repeat in range(N_REPEATS):
            print(f"\n  repeat {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat

            # Create missing (in the Test section only)
            test_data = target_data.loc[test_idx].copy()

            masked_test, missing_indices_local = create_random_clustered_missing(
                test_data, missing_ratio, min_segment=5, max_segment=30, seed=seed
            )

            missing_indices_global = test_idx[missing_indices_local]

            # Build a comprehensive dataset
            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # Feature construction
            df_features = feature_engineer.create_features(
                full_target, neighbor_data,
                n_lags=7, rolling_windows=[3, 7, 14],
                include_target_lags=True, use_neighbor_lag0=True
            )

            # === Full Model ===
            interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)

            method_results['lgbm_full']['rmse'].append(metrics['rmse'])
            method_results['lgbm_full']['mae'].append(metrics['mae'])
            method_results['lgbm_full']['corr'].append(metrics['correlation'])
            method_results['lgbm_full']['ci_cov'].append(metrics['ci_coverage'])

            # === Non-spatial model  ===
            df_no_spatial = df_features.drop(columns=[
                                                         'neighbor_r', 'neighbor_dist_km', 'spatial_weight'
                                                     ] + [c for c in df_features.columns if 'nbr_adj' in c],
                                             errors='ignore')

            train_df_ns = df_no_spatial.loc[train_idx.union(val_idx)].dropna(subset=['target'])
            X_train_ns = train_df_ns.drop(columns=['target', 'neighbor'], errors='ignore').fillna(train_df_ns.mean())
            y_train_ns = train_df_ns['target']

            model_ns = lgb.LGBMRegressor(**LGBM_PARAMS)
            model_ns.fit(X_train_ns, y_train_ns)

            X_test_ns = df_no_spatial.loc[missing_indices_global].drop(columns=['target', 'neighbor'],
                                                                       errors='ignore').fillna(X_train_ns.mean())
            pred_ns = model_ns.predict(X_test_ns)

            metrics_ns = calculate_metrics_with_ci(
                target_data, pred_ns, None, None, missing_indices_global
            )

            method_results['lgbm_no_spatial']['rmse'].append(metrics_ns['rmse'])
            method_results['lgbm_no_spatial']['mae'].append(metrics_ns['mae'])
            method_results['lgbm_no_spatial']['corr'].append(metrics_ns['correlation'])

            # === Baseline ===
            pred_time = BaselineInterpolators.time_only_baseline(
                df_features, train_idx.union(val_idx), missing_indices_global
            )
            metrics_time = calculate_metrics_with_ci(
                target_data, pred_time, None, None, missing_indices_global
            )
            method_results['time_only']['rmse'].append(metrics_time['rmse'])
            method_results['time_only']['mae'].append(metrics_time['mae'])
            method_results['time_only']['corr'].append(metrics_time['correlation'])

            method_results['linear']['rmse'].append(metrics_linear['rmse'])
            method_results['linear']['mae'].append(metrics_linear['mae'])
            method_results['linear']['corr'].append(metrics_linear['correlation'])

            print(f"    ✓ LGBM_Full: RMSE={metrics['rmse']:.4f}, CI_cov={metrics['ci_coverage']:.3f}")

       
        for method_name, method_data in method_results.items():
            if len(method_data['rmse']) > 0:
                results_list.append({
                    'experiment': 'random',
                    'station': TARGET_STATION,
                    'missing_ratio': missing_ratio * 100,
                    'method': method_name,
                    'rmse_mean': np.nanmean(method_data['rmse']),
                    'rmse_std': np.nanstd(method_data['rmse']),
                    'mae_mean': np.nanmean(method_data['mae']),
                    'mae_std': np.nanstd(method_data['mae']),
                    'correlation_mean': np.nanmean(method_data['corr']),
                    'correlation_std': np.nanstd(method_data['corr']),
                    'ci_coverage_mean': np.nanmean(method_data.get('ci_cov', [np.nan])),
                    'n_repeats': N_REPEATS
                })

    # Save result
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(output_folder, 'lgbm_spatial_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ The results have been saved: {results_path}")

    return results_df, interpolator


def plot_feature_importance(interpolator, output_folder):
    """Plotting Feature Importance"""
    importance_df = interpolator.get_feature_importance(top_n=20)

    if importance_df is None:
        return

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'].values)
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Feature Importance', fontsize=13)
    plt.title('LightGBM Feature Importance (Top 20)', fontsize=15, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ The feature importance plot has been saved: {save_path}")
    plt.close()


def plot_comparison_results(results_df, output_folder):
    """Comparison Chart of Drawing Methods"""

    # Figure 1: Comparison of Continuous Missing Data and RMSE
    continuous_df = results_df[results_df['experiment'] == 'continuous']

    if len(continuous_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE
        pivot_rmse = continuous_df.pivot_table(
            values='rmse_mean', index='missing_days', columns='method'
        )

        ax = axes[0]
        for method in pivot_rmse.columns:
            ax.plot(pivot_rmse.index, pivot_rmse[method], marker='o', label=method, linewidth=2)

        ax.set_xlabel('Missing Days', fontsize=12)
        ax.set_ylabel('RMSE (mm)', fontsize=12)
        ax.set_title('Continuous Missing - RMSE', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MAE
        pivot_mae = continuous_df.pivot_table(
            values='mae_mean', index='missing_days', columns='method'
        )

        ax = axes[1]
        for method in pivot_mae.columns:
            ax.plot(pivot_mae.index, pivot_mae[method], marker='s', label=method, linewidth=2)

        ax.set_xlabel('Missing Days', fontsize=12)
        ax.set_ylabel('MAE (mm)', fontsize=12)
        ax.set_title('Continuous Missing - MAE', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_folder, 'continuous_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ The comparison chart of consecutive missing values has been saved: {save_path}")
        plt.close()

    # Figure 2: Comparison of Random Missing Data and RMSE
    random_df = results_df[results_df['experiment'] == 'random']

    if len(random_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pivot_rmse = random_df.pivot_table(
            values='rmse_mean', index='missing_ratio', columns='method'
        )

        ax = axes[0]
        for method in pivot_rmse.columns:
            ax.plot(pivot_rmse.index, pivot_rmse[method], marker='o', label=method, linewidth=2)

        ax.set_xlabel('Missing Ratio (%)', fontsize=12)
        ax.set_ylabel('RMSE (mm)', fontsize=12)
        ax.set_title('Random Missing - RMSE', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        pivot_mae = random_df.pivot_table(
            values='mae_mean', index='missing_ratio', columns='method'
        )

        ax = axes[1]
        for method in pivot_mae.columns:
            ax.plot(pivot_mae.index, pivot_mae[method], marker='s', label=method, linewidth=2)

        ax.set_xlabel('Missing Ratio (%)', fontsize=12)
        ax.set_ylabel('MAE (mm)', fontsize=12)
        ax.set_title('Random Missing - MAE', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_folder, 'random_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ The comparison chart for missing at random has been saved: {save_path}")
        plt.close()


def generate_summary_report(results_df, output_folder):
    """Generate a summary report"""
    print("\n" + "=" * 80)
    print("📊 Generate a summary report")
    print("=" * 80)

    # 1. Overall Performance
    summary_overall = results_df.groupby('method').agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean',
        'ci_coverage_mean': 'mean'
    }).round(4)

    print("\nOverall Average Performance:")
    print(summary_overall)

    overall_path = os.path.join(output_folder, 'summary_overall.csv')
    summary_overall.to_csv(overall_path, encoding='utf-8-sig')

    # 2. Summary by Experiment Type
    summary_by_exp = results_df.groupby(['experiment', 'method']).agg({
        'rmse_mean': ['mean', 'std'],
        'mae_mean': ['mean', 'std'],
        'correlation_mean': ['mean', 'std']
    }).round(4)

    print("\nSummary by Experiment Type:")
    print(summary_by_exp)

    exp_path = os.path.join(output_folder, 'summary_by_experiment.csv')
    summary_by_exp.to_csv(exp_path, encoding='utf-8-sig')

    
    # 3. Percentage improvement
    if 'lgbm_full' in ranking.index and 'linear' in ranking.index:
        improvement = (ranking['linear'] - ranking['lgbm_full']) / ranking['linear'] * 100
        print(f"\n✨ LGBM_Full Compared to Linear Improvements: {improvement:.2f}%")

    if 'lgbm_full' in ranking.index and 'lgbm_no_spatial' in ranking.index:
        spatial_benefit = (ranking['lgbm_no_spatial'] - ranking['lgbm_full']) / ranking['lgbm_no_spatial'] * 100
        print(f"✨ Contribution of spatial characteristics: {spatial_benefit:.2f}%")


# =====================================================================
# 主程序
# =====================================================================
if __name__ == "__main__":
    print(“\n” + “=” * 80)
    print(“LightGBM + Spatial Correlation Fusion Interpolation System”)
    print(“=” * 80)
    print(f“Configuration Information:”)
    print(f“  - Target Station: {TARGET_STATION}”)
    print(f“  - Neighboring Station: {NEIGHBOR_STATION}”)
    print(f“  - Spatial Correlation: r = {CORRELATION_R:.4f}”)
    print(f“  - Station Distance: {NEIGHBOR_DIST_KM:.2f} km”)
    print(f“  - Analysis period: {START_DATE} to {END_DATE}”)
    print(f“  - Number of repetitions: {N_REPEATS}”)
    print(“=” * 80)

    # 1. Load data
    print(“\nLoading data...”)
    try:
        df_target = pd.read_csv(
            os.path.join(FOLDER, f"{TARGET_STATION}.csv"),
            parse_dates=[TIME_COL]
        )
        df_neighbor = pd.read_csv(
            os.path.join(FOLDER, f"{NEIGHBOR_STATION}.csv"),
            parse_dates=[TIME_COL]
        )

        # Process data
        df_target = df_target.set_index(TIME_COL).sort_index()
        df_neighbor = df_neighbor.set_index(TIME_COL).sort_index()

        # Extract data for a specified time period
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)

        target_data = df_target.loc[start_dt:end_dt, VALUE_COL]
        neighbor_data = df_neighbor.loc[start_dt:end_dt, VALUE_COL]

        # Reindexing ensures continuity
        full_date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        target_data = target_data.reindex(full_date_range)
        neighbor_data = neighbor_data.reindex(full_date_range)

        print(f"✓ Data loaded successfully")
        print(f"  - target station: {len(target_data)} days, missing {target_data.isna().sum()} days")
        print(f"  - neighbor station: {len(neighbor_data)} days, missing {neighbor_data.isna().sum()} 天")

    except Exception as e:
        print(f"[Error] Data loading failed: {e}")
        exit(1)

    # 2. Create an output folder
    output_folder = os.path.join(FOLDER, "lgbm_spatial_results")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder: {output_folder}")
    # ===================================================================
    # 🔥 Hyperparameter Optimization
    # ===================================================================
    ENABLE_HYPERPARAMETER_OPTIMIZATION = True  

    if ENABLE_HYPERPARAMETER_OPTIMIZATION:
        print("\n" + "=" * 80)
        print("⚙️ Start hyperparameter optimization")
        print("=" * 80)

        best_params = optimize_lgbm_hyperparameters(
            target_data, neighbor_data, output_folder,
            n_trials=100  
        )

        # Update global parameters
        LGBM_PARAMS.update(best_params)

        # Save optimal parameters
        best_params_path = os.path.join(output_folder, 'best_hyperparameters.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4, ensure_ascii=False)

         print(f“\n✓ Optimal hyperparameters saved: {best_params_path}”)
        print(“\n💡 Tip: For subsequent runs, set ENABLE_HYPERPARAMETER_OPTIMIZATION to False”)
        print(“         and manually update the LGBM_PARAMS dictionary to save time”)
    else:
        print("\n⏭️ Skip hyperparameter tuning and use preset parameters")

    # ===================================================================
    # 3. Determine the experimental parameters
    sample_length = len(target_data)

    if sample_length > 1000:
        missing_days_list = [7, 15, 30, 60, 90, 120, 180]
    elif sample_length > 500:
        missing_days_list = [7, 15, 30, 60]
    else:
        missing_days_list = [7, 15, 30]

    missing_ratios = np.arange(0.05, 0.55, 0.05)

    print(f“\nExperiment configuration:”)
    print(f“  - Number of consecutive missing days: {missing_days_list}”)
    print(f“  - Random missing rates: {[f'{r * 100:.0f}%' for r in missing_ratios]}”)

    # 4. Conduct the experiment
    results_df, final_interpolator = run_spatial_lgbm_experiments(
        target_data, neighbor_data,
        missing_days_list, missing_ratios,
        output_folder
    )

    # 5. 生成可视化
    print("\n" + "=" * 80)
    print("Generate visualizations...")
    print("=" * 80)

    plot_feature_importance(final_interpolator, output_folder)
    plot_comparison_results(results_df, output_folder)

    # 6. Generate a summary report
    generate_summary_report(results_df, output_folder)

    # 7. Save configuration settings
    config_info = {
        'target_station': TARGET_STATION,
        'neighbor_station': NEIGHBOR_STATION,
        'correlation_r': CORRELATION_R,
        'neighbor_dist_km': NEIGHBOR_DIST_KM,
        'bias_value': BIAS_VALUE,
        'time_range': f"{START_DATE} to {END_DATE}",
        'data_length': sample_length,
        'n_repeats': N_REPEATS,
        'n_bootstrap': N_BOOTSTRAP,
        'lgbm_params': LGBM_PARAMS,
        'missing_days_list': missing_days_list,
        'missing_ratios': missing_ratios.tolist(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    config_path = os.path.join(output_folder, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(“✅ Experiment complete!”)
    print(“=” * 80)
    print(f“All results have been saved to: {output_folder}”)
    print(f“  - Detailed results: lgbm_spatial_results.csv”)
    print(f“  - Overall summary: summary_overall.csv”)
    print(f“  - Experiment summary: summary_by_experiment.csv”)
    print(f“  - Feature importance plot: feature_importance.png”)
    print(f“  - Continuous vs. missing comparison: continuous_comparison.png”)
    print(f“  - Random vs. missing comparison: random_comparison.png”)
    print(f“  - Experiment configuration: experiment_config.json”)
    print("=" * 80 + "\n")
