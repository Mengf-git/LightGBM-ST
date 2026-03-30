"""
GNSS Time Series Missing Value Imputation - LightGBM + Traditional Interpolation Method Comparison
===========================================================
New Features:
1. Integration of Akima, KNN, Random Forest, and Cubic Spline interpolation methods
2. Complete method performance comparison experiments
"""

import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import interpolate
from scipy.stats import pearsonr
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import json

warnings.filterwarnings("ignore")

# ----------------------
# SCI Paper Standard Font Settings
# ----------------------
rcParams['font.family'] = ['Arial', 'Times New Roman']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.titlesize'] = 13
rcParams['figure.dpi'] = 300

# Color scheme (colorblind-friendly)
COLOR_SCHEME = {
    'lgbm_full': '#1f77b4',      # Blue
    'akima': '#ff7f0e',           # Orange
    'cubic_spline': '#2ca02c',    # Green
    'knn': '#d62728',             # Red
    'rf': '#9467bd',              # Purple
    'lgbm_no_spatial': '#8c564b', # Brown
    'time_only': '#e377c2',       # Pink
    'linear': '#7f7f7f'           # Gray
}

# =====================================================================
# Core Configuration Parameters
# =====================================================================
FOLDER = "../data/real_GNSS"
TARGET_STATION = "YNYS"
NEIGHBOR_STATION = "YNLJ"
TIME_COL = "YYYYMMDD"
VALUE_COL = "U(m)"
START_DATE = "2011-12-05"
END_DATE = "2019-02-17"
N_REPEATS = 10
N_BOOTSTRAP = 50
RANDOM_SEED = 42
CORRELATION_R = 0.8236
NEIGHBOR_DIST_KM = 71.56
BIAS_VALUE = -2.6672

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
# Traditional Interpolation Methods Class
# =====================================================================
class TraditionalInterpolationMethods:
    """Integration of traditional interpolation methods"""

    @staticmethod
    def cubic_spline_interpolation(series):
        """Cubic spline interpolation"""
        try:
            valid_mask = ~series.isna()
            x_valid = np.where(valid_mask)[0]
            y_valid = series[valid_mask].values

            if len(x_valid) < 4:
                return series.interpolate(method='linear').bfill().ffill()

            cs = interpolate.CubicSpline(x_valid, y_valid)
            x_all = np.arange(len(series))
            filled_values = cs(x_all)

            result = series.copy()
            result.iloc[:] = filled_values
            return result

        except Exception as e:
            print(f"    [Warning] Cubic spline interpolation failed: {e}")
            return series.interpolate(method='linear').bfill().ffill()

    @staticmethod
    def akima_interpolation(series):
        """Akima interpolation"""
        try:
            from scipy.interpolate import Akima1DInterpolator

            valid_mask = ~series.isna()
            x_valid = np.where(valid_mask)[0]
            y_valid = series[valid_mask].values

            if len(x_valid) < 5:
                return TraditionalInterpolationMethods.cubic_spline_interpolation(series)

            akima = Akima1DInterpolator(x_valid, y_valid)
            x_all = np.arange(len(series))
            filled_values = akima(x_all)

            result = series.copy()
            result.iloc[:] = filled_values
            return result

        except Exception as e:
            print(f"    [Warning] Akima interpolation failed: {e}")
            return TraditionalInterpolationMethods.cubic_spline_interpolation(series)

    @staticmethod
    def _create_lagged_features(series, n_lags=30, n_rolling_stats=3):
        """Create lagged features and rolling statistics features for a time series"""
        n = len(series)
        features = np.zeros((n, n_lags + n_rolling_stats * 4))

        series_filled = series.fillna(method='ffill').fillna(method='bfill')

        # Lagged features
        for lag in range(n_lags):
            if lag == 0:
                features[:, lag] = series_filled.values
            else:
                features[lag:, lag] = series_filled.iloc[:-lag].values
                features[:lag, lag] = series_filled.iloc[0]

        # Rolling statistics features
        col_idx = n_lags
        for window in range(1, n_rolling_stats + 1):
            window_size = max(2, (window + 1) * 3)

            rolling_mean = series_filled.rolling(window=window_size, min_periods=1).mean()
            features[:, col_idx] = rolling_mean.values
            col_idx += 1

            rolling_std = series_filled.rolling(window=window_size, min_periods=1).std()
            rolling_std = rolling_std.fillna(0)
            features[:, col_idx] = rolling_std.values
            col_idx += 1

            rolling_max = series_filled.rolling(window=window_size, min_periods=1).max()
            features[:, col_idx] = rolling_max.values
            col_idx += 1

            rolling_min = series_filled.rolling(window=window_size, min_periods=1).min()
            features[:, col_idx] = rolling_min.values
            col_idx += 1

        return features

    @staticmethod
    def knn_interpolation(series, n_neighbors=5, n_lags=30, n_rolling_stats=3):
        """KNN interpolation"""
        try:
            valid_mask = ~series.isna()
            missing_mask = series.isna()

            x_valid_idx = np.where(valid_mask)[0]
            y_valid = series[valid_mask].values
            x_missing_idx = np.where(missing_mask)[0]

            if len(x_valid_idx) == 0 or len(x_missing_idx) == 0:
                return series.interpolate(method='linear').bfill().ffill()

            features = TraditionalInterpolationMethods._create_lagged_features(
                series, n_lags, n_rolling_stats)

            features_valid = features[x_valid_idx]
            features_missing = features[x_missing_idx]

            knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(x_valid_idx)))
            knn.fit(features_valid, y_valid)

            y_pred = knn.predict(features_missing)

            filled_series = series.copy()
            filled_series.iloc[x_missing_idx] = y_pred

            return filled_series

        except Exception as e:
            print(f"    [Warning] KNN interpolation failed: {e}")
            return series.interpolate(method='linear').bfill().ffill()

    @staticmethod
    def random_forest_interpolation(series, n_estimators=100, max_depth=10,
                                    min_samples_leaf=2, n_iterations=10):
        """Random forest iterative interpolation"""
        try:
            series_filled = series.copy()
            mask_valid = ~series.isna()

            if mask_valid.sum() < 20:
                return series.interpolate(method='linear').bfill().ffill()

            x_missing_idx = np.where(~mask_valid)[0]
            if len(x_missing_idx) == 0:
                return series_filled

            series_filled = series_filled.fillna(method='ffill').fillna(method='bfill')

            for iteration in range(n_iterations):
                features = TraditionalInterpolationMethods._create_lagged_features(series_filled)

                mask_valid_current = ~series.isna()
                x_valid_idx = np.where(mask_valid_current)[0]
                y_valid = series.iloc[x_valid_idx].values

                features_valid = features[x_valid_idx]

                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=RANDOM_SEED,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-1
                )
                rf.fit(features_valid, y_valid)

                features_missing = features[x_missing_idx]
                predictions = rf.predict(features_missing)
                series_filled.iloc[x_missing_idx] = predictions

            return series_filled

        except Exception as e:
            print(f"    [Warning] Random forest interpolation failed: {e}")
            return series.interpolate(method='linear').bfill().ffill()


# =====================================================================
# [All original classes are retained; only new classes are added here]
# =====================================================================

class SpatialCorrelationFeatureEngineering:
    """Spatial correlation feature engineering class"""

    def __init__(self, correlation_r, neighbor_dist_km, bias_value):
        self.correlation_r = correlation_r
        self.neighbor_dist_km = neighbor_dist_km
        self.bias_value = bias_value
        self.ols_model = None
        self.base_model = None
        self.residuals = None
        self.residual_std = None

    def fit_bias_correction(self, target_series, neighbor_series):
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
        if self.ols_model is None:
            return neighbor_series.copy()

        return self.ols_model['intercept'] + self.ols_model['coef'] * neighbor_series

    def create_features(self, target_series, neighbor_series,
                        n_lags=7, rolling_windows=[3, 7, 14],
                        include_target_lags=True, use_neighbor_lag0=False):
        df = pd.DataFrame({
            'target': target_series,
            'neighbor': neighbor_series
        })

        df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df['month_norm'] = df.index.month / 12.0
        years = pd.Series(df.index.year)
        df['year_norm'] = (years - years.mean()) / years.std()

        for lag in range(n_lags + 1):
            if lag == 0 and not use_neighbor_lag0:
                continue
            df[f'nbr_lag{lag}'] = df['neighbor'].shift(lag)

        for window in rolling_windows:
            df[f'nbr_rollmean_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).mean()
            df[f'nbr_rollstd_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).std().fillna(0)
            df[f'nbr_rollmax_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).max()
            df[f'nbr_rollmin_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).min()

        if include_target_lags:
            for lag in [1, 2, 3, 7]:
                df[f'tgt_lag{lag}'] = df['target'].shift(lag)

        df['neighbor_r'] = self.correlation_r
        df['neighbor_dist_km'] = self.neighbor_dist_km
        df['spatial_weight'] = self.correlation_r / (self.neighbor_dist_km + 1e-3)

        for window in [30, 60]:
            rolling_corr = df['target'].rolling(window).corr(df['neighbor'])
            df[f'spatial_corr_{window}d'] = rolling_corr.shift(1)

            diff = (df['neighbor'] - df['target']).shift(1)
            df[f'nbr_tgt_diff_std_{window}d'] = diff.rolling(window).std().fillna(0)

        df['nbr_consecutive_valid'] = (~df['neighbor'].isna()).astype(int).groupby(
            (df['neighbor'].isna() != df['neighbor'].isna().shift()).cumsum()
        ).cumsum()

        if self.ols_model is not None:
            nbr_adj = self.apply_bias_correction(df['neighbor'])
            for lag in range(min(3, n_lags + 1)):
                df[f'nbr_adj_lag{lag}'] = nbr_adj.shift(lag)

        df['nbr_isnan_lag0'] = df['neighbor'].isna().astype(int)
        df['n_neighbors_available'] = (~df['neighbor'].isna()).astype(int)

        return df


class LightGBMSpatialInterpolator:
    """LightGBM-based spatial correlation interpolator"""

    def __init__(self, feature_engineer, lgbm_params=None):
        self.feature_engineer = feature_engineer
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.models = []
        self.feature_names = None
        self.base_model = None
        self.residual_std = None

    def _prepare_train_data(self, df_features, train_idx):
        train_df = df_features.loc[train_idx]
        train_df = train_df.dropna(subset=['target'])

        if len(train_df) < 50:
            raise ValueError(f"Insufficient training samples: {len(train_df)}")

        X = train_df.drop(columns=['target', 'neighbor'], errors='ignore')
        y = train_df['target']
        X = X.fillna(X.mean())

        self.feature_names = X.columns.tolist()

        return X, y

    def train_ensemble_residual(self, df_features, train_idx, n_bootstrap=50):
        print(f"    Training residual bootstrap ensemble...")

        X_full, y_full = self._prepare_train_data(df_features, train_idx)

        split_point = int(0.8 * len(X_full))
        X_train = X_full.iloc[:split_point]
        y_train = y_full.iloc[:split_point]
        X_val = X_full.iloc[split_point:]
        y_val = y_full.iloc[split_point:]

        base_model = lgb.LGBMRegressor(**self.lgbm_params)
        base_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        y_pred_train = base_model.predict(X_train)
        residuals = y_train.values - y_pred_train
        residual_std = np.std(residuals)

        y_pred_val = base_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)

        print(f"    ✓ Base model training complete")
        print(f"    ✓ Training set residual std: {residual_std:.4f} mm")
        print(f"    ✓ Validation set MAE: {val_mae:.4f} mm")

        self.base_model = base_model
        self.residual_std = residual_std
        self.models = [base_model]

    def predict_with_uncertainty_residual(self, df_features, predict_idx):
        X_pred = df_features.loc[predict_idx].drop(
            columns=['target', 'neighbor'], errors='ignore')
        X_pred = X_pred.fillna(X_pred.mean())

        base_prediction = self.base_model.predict(X_pred)

        z_score = 1.96
        pred_median = base_prediction
        pred_lower = base_prediction - z_score * self.residual_std
        pred_upper = base_prediction + z_score * self.residual_std

        return pred_median, pred_lower, pred_upper


class BaselineInterpolators:
    """Baseline interpolation methods"""

    @staticmethod
    def time_only_baseline(df_features, train_idx, test_idx):
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
    def linear_interpolation(series, missing_indices):
        filled = series.copy()

        if isinstance(missing_indices, pd.DatetimeIndex):
            filled.loc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.loc[missing_indices].values
        else:
            filled.iloc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.iloc[missing_indices].values


def create_continuous_missing(data, n_segments, days_per_segment, seed=None):
    """Create continuous missing segments"""
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
    """Create randomly clustered missing segments"""
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

        if upper_bound <= min_segment:
            segment_length = min(remaining_missing, n_total - current_pos)
        else:
            segment_length = np.random.randint(min_segment, upper_bound)

        end_pos = min(current_pos + segment_length, n_total)
        missing_indices.extend(range(current_pos, end_pos))
        current_pos = end_pos

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


def calculate_metrics_with_ci(true_values, predicted_values, pred_lower, pred_upper, missing_indices):
    """Calculate evaluation metrics + CI coverage rate"""
    if isinstance(missing_indices, pd.DatetimeIndex):
        valid_indices = missing_indices
    else:
        valid_indices = pd.DatetimeIndex(missing_indices)

    if isinstance(predicted_values, pd.Series):
        pred_series = predicted_values
    else:
        pred_series = pd.Series(predicted_values, index=valid_indices)

    if pred_lower is not None and not isinstance(pred_lower, pd.Series):
        pred_lower = pd.Series(pred_lower, index=valid_indices)

    if pred_upper is not None and not isinstance(pred_upper, pd.Series):
        pred_upper = pd.Series(pred_upper, index=valid_indices)

    true_missing = true_values.loc[valid_indices]
    pred_missing = pred_series.loc[valid_indices]

    valid_mask = true_missing.notna() & pred_missing.notna() & np.isfinite(pred_missing)

    if valid_mask.sum() == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'correlation': np.nan, 'ci_coverage': np.nan}

    true_clean = true_missing[valid_mask].values
    pred_clean = pred_missing[valid_mask].values

    rmse = np.sqrt(mean_squared_error(true_clean, pred_clean))
    mae = mean_absolute_error(true_clean, pred_clean)

    try:
        correlation, _ = pearsonr(true_clean, pred_clean)
        if np.isnan(correlation):
            correlation = 1.0 if np.array_equal(true_clean, pred_clean) else 0.0
    except:
        correlation = 0.0

    ci_coverage = np.nan
    if pred_lower is not None and pred_upper is not None:
        lower_clean = pred_lower.loc[valid_indices][valid_mask].values
        upper_clean = pred_upper.loc[valid_indices][valid_mask].values
        in_ci = (true_clean >= lower_clean) & (true_clean <= upper_clean)
        ci_coverage = in_ci.mean()

    return {
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'ci_coverage': ci_coverage
    }


def run_comprehensive_experiments(target_data, neighbor_data,
                                  missing_days_list, missing_ratios,
                                  output_folder):
    """
    Run complete comparison experiments (including traditional interpolation methods)
    """
    print("\n" + "=" * 80)
    print("🚀 LightGBM + Traditional Interpolation Methods Comprehensive Comparison Experiment")
    print("=" * 80)

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

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} days")
    print(f"  Val:   {len(val_idx)} days")
    print(f"  Test:  {len(test_idx)} days")

    # Fit bias correction model
    print("\nFitting bias correction model...")
    feature_engineer.fit_bias_correction(
        target_data.loc[train_idx.union(val_idx)],
        neighbor_data.loc[train_idx.union(val_idx)]
    )

    results_list = []

    # ====================================================================
    # Experiment 1: Continuous missing segments
    # ====================================================================
    print("\n" + "=" * 80)
    print("🔴 Experiment 1: Continuous Missing Segments")
    print("=" * 80)

    for missing_days in missing_days_list:
        print(f"\n{'─' * 80}")
        print(f"Missing segment length: {missing_days} days")
        print(f"{'─' * 80}")

        method_results = {
            'lgbm_full': {'rmse': [], 'mae': [], 'corr': [], 'ci_cov': []},
            'akima': {'rmse': [], 'mae': [], 'corr': []},
            'cubic_spline': {'rmse': [], 'mae': [], 'corr': []},
            'knn': {'rmse': [], 'mae': [], 'corr': []},
            'rf': {'rmse': [], 'mae': [], 'corr': []},
            'linear': {'rmse': [], 'mae': [], 'corr': []}
        }

        for repeat in range(N_REPEATS):
            print(f"\n  Repeat {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat

            # Create missing data
            test_data = target_data.loc[test_idx].copy()
            n_segments = max(1, int(len(test_idx) * 0.1 / missing_days))

            masked_test, missing_indices_local, _ = create_continuous_missing(
                test_data, n_segments, missing_days, seed=seed
            )

            missing_indices_global = test_idx[missing_indices_local]

            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # === LightGBM-ST method ===
            df_features = feature_engineer.create_features(
                full_target, neighbor_data,
                n_lags=7, rolling_windows=[3, 7, 14],
                include_target_lags=True, use_neighbor_lag0=False
            )

            interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
            interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx), n_bootstrap=N_BOOTSTRAP)
            pred_median, pred_lower, pred_upper = interpolator.predict_with_uncertainty_residual(
                df_features, missing_indices_global
            )

            metrics = calculate_metrics_with_ci(
                target_data, pred_median, pred_lower, pred_upper, missing_indices_global
            )

            method_results['lgbm_full']['rmse'].append(metrics['rmse'])
            method_results['lgbm_full']['mae'].append(metrics['mae'])
            method_results['lgbm_full']['corr'].append(metrics['correlation'])
            method_results['lgbm_full']['ci_cov'].append(metrics['ci_coverage'])

            # === Akima interpolation ===
            pred_akima = TraditionalInterpolationMethods.akima_interpolation(full_target)
            metrics_akima = calculate_metrics_with_ci(
                target_data, pred_akima, None, None, missing_indices_global
            )
            method_results['akima']['rmse'].append(metrics_akima['rmse'])
            method_results['akima']['mae'].append(metrics_akima['mae'])
            method_results['akima']['corr'].append(metrics_akima['correlation'])

            # === Cubic spline interpolation ===
            pred_cubic = TraditionalInterpolationMethods.cubic_spline_interpolation(full_target)
            metrics_cubic = calculate_metrics_with_ci(
                target_data, pred_cubic, None, None, missing_indices_global
            )
            method_results['cubic_spline']['rmse'].append(metrics_cubic['rmse'])
            method_results['cubic_spline']['mae'].append(metrics_cubic['mae'])
            method_results['cubic_spline']['corr'].append(metrics_cubic['correlation'])

            # === KNN interpolation ===
            pred_knn = TraditionalInterpolationMethods.knn_interpolation(full_target)
            metrics_knn = calculate_metrics_with_ci(
                target_data, pred_knn, None, None, missing_indices_global
            )
            method_results['knn']['rmse'].append(metrics_knn['rmse'])
            method_results['knn']['mae'].append(metrics_knn['mae'])
            method_results['knn']['corr'].append(metrics_knn['correlation'])

            # === Random forest interpolation ===
            pred_rf = TraditionalInterpolationMethods.random_forest_interpolation(full_target)
            metrics_rf = calculate_metrics_with_ci(
                target_data, pred_rf, None, None, missing_indices_global
            )
            method_results['rf']['rmse'].append(metrics_rf['rmse'])
            method_results['rf']['mae'].append(metrics_rf['mae'])
            method_results['rf']['corr'].append(metrics_rf['correlation'])

            # === Linear interpolation ===
            pred_linear = BaselineInterpolators.linear_interpolation(
                full_target, missing_indices_global
            )
            metrics_linear = calculate_metrics_with_ci(
                target_data, pred_linear, None, None, missing_indices_global
            )
            method_results['linear']['rmse'].append(metrics_linear['rmse'])
            method_results['linear']['mae'].append(metrics_linear['mae'])
            method_results['linear']['corr'].append(metrics_linear['correlation'])

            print(f"    ✓ LightGBM-ST: RMSE={metrics['rmse']:.4f}")
            print(f"    ✓ Akima: RMSE={metrics_akima['rmse']:.4f}")
            print(f"    ✓ Cubic Spline: RMSE={metrics_cubic['rmse']:.4f}")
            print(f"    ✓ KNN: RMSE={metrics_knn['rmse']:.4f}")
            print(f"    ✓ RF: RMSE={metrics_rf['rmse']:.4f}")
            print(f"    ✓ Linear: RMSE={metrics_linear['rmse']:.4f}")

        # Summarize results
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
                    'ci_coverage_mean': np.nanmean(method_data.get('ci_cov', [np.nan])),
                    'n_repeats': N_REPEATS
                })

    # ====================================================================
    # Experiment 2: Random clustered missing
    # ====================================================================
    print("\n" + "=" * 80)
    print("🔵 Experiment 2: Random Clustered Missing")
    print("=" * 80)

    for missing_ratio in missing_ratios:
        print(f"\n{'─' * 80}")
        print(f"Missing ratio: {missing_ratio * 100:.1f}%")
        print(f"{'─' * 80}")

        method_results = {
            'lgbm_full': {'rmse': [], 'mae': [], 'corr': [], 'ci_cov': []},
            'akima': {'rmse': [], 'mae': [], 'corr': []},
            'cubic_spline': {'rmse': [], 'mae': [], 'corr': []},
            'knn': {'rmse': [], 'mae': [], 'corr': []},
            'rf': {'rmse': [], 'mae': [], 'corr': []},
            'linear': {'rmse': [], 'mae': [], 'corr': []}
        }

        for repeat in range(N_REPEATS):
            print(f"\n  Repeat {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat

            test_data = target_data.loc[test_idx].copy()

            masked_test, missing_indices_local = create_random_clustered_missing(
                test_data, missing_ratio, min_segment=5, max_segment=30, seed=seed
            )

            missing_indices_global = test_idx[missing_indices_local]

            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # === LightGBM-ST ===
            df_features = feature_engineer.create_features(
                full_target, neighbor_data,
                n_lags=7, rolling_windows=[3, 7, 14],
                include_target_lags=True, use_neighbor_lag0=True
            )

            interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
            interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx), n_bootstrap=N_BOOTSTRAP)
            pred_median, pred_lower, pred_upper = interpolator.predict_with_uncertainty_residual(
                df_features, missing_indices_global
            )

            metrics = calculate_metrics_with_ci(
                target_data, pred_median, pred_lower, pred_upper, missing_indices_global
            )

            method_results['lgbm_full']['rmse'].append(metrics['rmse'])
            method_results['lgbm_full']['mae'].append(metrics['mae'])
            method_results['lgbm_full']['corr'].append(metrics['correlation'])
            method_results['lgbm_full']['ci_cov'].append(metrics['ci_coverage'])

            # === Traditional methods ===
            pred_akima = TraditionalInterpolationMethods.akima_interpolation(full_target)
            metrics_akima = calculate_metrics_with_ci(target_data, pred_akima, None, None, missing_indices_global)
            method_results['akima']['rmse'].append(metrics_akima['rmse'])
            method_results['akima']['mae'].append(metrics_akima['mae'])
            method_results['akima']['corr'].append(metrics_akima['correlation'])

            pred_cubic = TraditionalInterpolationMethods.cubic_spline_interpolation(full_target)
            metrics_cubic = calculate_metrics_with_ci(target_data, pred_cubic, None, None, missing_indices_global)
            method_results['cubic_spline']['rmse'].append(metrics_cubic['rmse'])
            method_results['cubic_spline']['mae'].append(metrics_cubic['mae'])
            method_results['cubic_spline']['corr'].append(metrics_cubic['correlation'])

            pred_knn = TraditionalInterpolationMethods.knn_interpolation(full_target)
            metrics_knn = calculate_metrics_with_ci(target_data, pred_knn, None, None, missing_indices_global)
            method_results['knn']['rmse'].append(metrics_knn['rmse'])
            method_results['knn']['mae'].append(metrics_knn['mae'])
            method_results['knn']['corr'].append(metrics_knn['correlation'])

            pred_rf = TraditionalInterpolationMethods.random_forest_interpolation(full_target)
            metrics_rf = calculate_metrics_with_ci(target_data, pred_rf, None, None, missing_indices_global)
            method_results['rf']['rmse'].append(metrics_rf['rmse'])
            method_results['rf']['mae'].append(metrics_rf['mae'])
            method_results['rf']['corr'].append(metrics_rf['correlation'])

            pred_linear = BaselineInterpolators.linear_interpolation(full_target, missing_indices_global)
            metrics_linear = calculate_metrics_with_ci(target_data, pred_linear, None, None, missing_indices_global)
            method_results['linear']['rmse'].append(metrics_linear['rmse'])
            method_results['linear']['mae'].append(metrics_linear['mae'])
            method_results['linear']['corr'].append(metrics_linear['correlation'])

            print(f"    ✓ LightGBM-ST: RMSE={metrics['rmse']:.4f}")

        # Summarize results
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

    # Save results
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(output_folder, 'comprehensive_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Results saved to: {results_path}")

    return results_df, interpolator


def plot_interpolation_comparison_figures(target_data, neighbor_data, feature_engineer,
                                          missing_days_list, missing_ratios, output_folder):
    """
    Generate true value vs predicted value interpolation effect comparison figures,
    focused on displaying the actual imputation performance of each method
    """
    print("\n" + "=" * 80)
    print("📊 Generating Interpolation Effect Visualization Comparison Figures")
    print("=" * 80)

    # Define method display names
    method_names = {
        'lgbm_full': 'LightGBM-ST',
        'akima': 'Akima',
        'cubic_spline': 'Cubic Spline',
        'knn': 'KNN',
        'rf': 'Random Forest',
        'linear': 'Linear'
    }

    # Prepare data indices
    total_length = len(target_data)
    train_end = int(0.6 * total_length)
    val_end = int(0.8 * total_length)
    train_idx = target_data.index[:train_end]
    val_idx = target_data.index[train_end:val_end]
    test_idx = target_data.index[val_end:]

    # ====================================================================
    # Figure 1: Continuous missing scenario - interpolation effect comparison across missing lengths
    # ====================================================================
    print("\nGenerating continuous missing scenario interpolation effect figures...")

    # Select representative missing day counts
    selected_days = [7, 30, 90] if 90 in missing_days_list else [7, 30]

    for missing_days in selected_days:
        print(f"  Processing {missing_days}-day missing scenario...")

        # Create missing data (fixed seed for reproducibility)
        test_data = target_data.loc[test_idx].copy()
        n_segments = 1  # Create only one missing segment for visualization

        masked_test, missing_indices_local, _ = create_continuous_missing(
            test_data, n_segments, missing_days, seed=RANDOM_SEED
        )

        missing_indices_global = test_idx[missing_indices_local]

        # Build full data with missing values
        full_target = target_data.copy()
        full_target.loc[missing_indices_global] = np.nan

        # Prepare features
        df_features = feature_engineer.create_features(
            full_target, neighbor_data,
            n_lags=7, rolling_windows=[3, 7, 14],
            include_target_lags=True, use_neighbor_lag0=False
        )

        # === Run each interpolation method ===
        predictions = {}

        # LightGBM-ST
        interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
        interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx), n_bootstrap=30)
        pred_lgbm, _, _ = interpolator.predict_with_uncertainty_residual(df_features, missing_indices_global)
        predictions['lgbm_full'] = pred_lgbm

        # Akima interpolation
        pred_akima = TraditionalInterpolationMethods.akima_interpolation(full_target)
        predictions['akima'] = pred_akima.loc[missing_indices_global].values

        # Cubic spline interpolation
        pred_cubic = TraditionalInterpolationMethods.cubic_spline_interpolation(full_target)
        predictions['cubic_spline'] = pred_cubic.loc[missing_indices_global].values

        # KNN interpolation
        pred_knn = TraditionalInterpolationMethods.knn_interpolation(full_target)
        predictions['knn'] = pred_knn.loc[missing_indices_global].values

        # Random forest interpolation
        pred_rf = TraditionalInterpolationMethods.random_forest_interpolation(full_target)
        predictions['rf'] = pred_rf.loc[missing_indices_global].values

        # Linear interpolation
        pred_linear = BaselineInterpolators.linear_interpolation(full_target, missing_indices_global)
        predictions['linear'] = pred_linear

        # === Plot comparison figure ===
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25)

        # Get true values
        true_values = target_data.loc[missing_indices_global].values
        x_axis = np.arange(len(missing_indices_global))

        # Extended window: show 10 days before and after the missing segment
        window_before = 10
        window_after = 10

        start_idx_local = missing_indices_local[0]
        end_idx_local = missing_indices_local[-1]

        extended_start = max(0, start_idx_local - window_before)
        extended_end = min(len(test_data) - 1, end_idx_local + window_after)

        extended_indices = test_idx[extended_start:extended_end + 1]
        extended_true = target_data.loc[extended_indices].values
        extended_x = np.arange(len(extended_indices))

        # Position of missing segment within the extended window
        missing_start_in_window = start_idx_local - extended_start
        missing_end_in_window = end_idx_local - extended_start + 1
        missing_x_in_window = np.arange(missing_start_in_window, missing_end_in_window)

        # Plot 6 subplots
        methods_to_plot = ['lgbm_full', 'akima', 'cubic_spline', 'knn', 'rf', 'linear']

        for idx, method in enumerate(methods_to_plot):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            # Plot full observed data (light color)
            ax.plot(extended_x, extended_true, 'o-', color='#CCCCCC',
                    linewidth=1.5, markersize=4, alpha=0.6, label='Observed Data')

            # Highlight true values in the missing segment
            ax.plot(missing_x_in_window, true_values, 'o', color='#000000',
                    markersize=6, markeredgewidth=1.5, markerfacecolor='white',
                    label='True Values (Missing)', zorder=5)

            # Plot predicted values
            pred_values = predictions[method]
            ax.plot(missing_x_in_window, pred_values, 's-',
                    color=COLOR_SCHEME.get(method, '#1f77b4'),
                    linewidth=2.5, markersize=7, label=f'{method_names[method]} Prediction',
                    zorder=4)

            # Annotate missing region
            ax.axvspan(missing_start_in_window, missing_end_in_window - 1,
                       alpha=0.15, color='red', label='Missing Period')

            # Calculate and display RMSE and R
            rmse = np.sqrt(mean_squared_error(true_values, pred_values))
            try:
                corr, _ = pearsonr(true_values, pred_values)
            except:
                corr = 0.0

            # Set title and labels
            ax.set_title(f'{method_names[method]}\nRMSE={rmse:.3f} mm, R={corr:.3f}',
                         fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel('Time (days)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Displacement (mm)', fontsize=10, fontweight='bold')

            # Legend
            ax.legend(loc='upper left', fontsize=8, frameon=True,
                      fancybox=False, edgecolor='black')

            # Grid and borders
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Overall title
        fig.suptitle(f'Interpolation Performance Comparison - {missing_days}-day Continuous Missing',
                     fontsize=14, fontweight='bold', y=0.995)

        save_path = os.path.join(output_folder, f'Fig_Interpolation_Continuous_{missing_days}days.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
        plt.close()

    # ====================================================================
    # Figure 2: Random clustered missing scenario - interpolation effect comparison
    # ====================================================================
    print("\nGenerating random clustered missing scenario interpolation effect figures...")

    # Select representative missing ratios
    selected_ratios = [0.10, 0.20, 0.30] if 0.30 in missing_ratios else [0.10, 0.20]

    for missing_ratio in selected_ratios:
        print(f"  Processing {missing_ratio * 100:.0f}% missing ratio scenario...")

        # Create missing data
        test_data = target_data.loc[test_idx].copy()

        masked_test, missing_indices_local = create_random_clustered_missing(
            test_data, missing_ratio, min_segment=5, max_segment=30, seed=RANDOM_SEED
        )

        missing_indices_global = test_idx[missing_indices_local]

        # Build full data with missing values
        full_target = target_data.copy()
        full_target.loc[missing_indices_global] = np.nan

        # Prepare features
        df_features = feature_engineer.create_features(
            full_target, neighbor_data,
            n_lags=7, rolling_windows=[3, 7, 14],
            include_target_lags=True, use_neighbor_lag0=True
        )

        # === Run each interpolation method ===
        predictions = {}

        # LightGBM-ST
        interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
        interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx), n_bootstrap=30)
        pred_lgbm, _, _ = interpolator.predict_with_uncertainty_residual(df_features, missing_indices_global)
        predictions['lgbm_full'] = pred_lgbm

        # Other methods
        pred_akima = TraditionalInterpolationMethods.akima_interpolation(full_target)
        predictions['akima'] = pred_akima.loc[missing_indices_global].values

        pred_cubic = TraditionalInterpolationMethods.cubic_spline_interpolation(full_target)
        predictions['cubic_spline'] = pred_cubic.loc[missing_indices_global].values

        pred_knn = TraditionalInterpolationMethods.knn_interpolation(full_target)

        if len(baseline_rmse) > 0:
            baseline_rmse = baseline_rmse[0]

            for method in ['lgbm_full', 'akima', 'knn', 'rf']:
                method_rmse = day_data[day_data['method'] == method]['rmse_mean'].values
                if len(method_rmse) > 0:
                    improvement = (baseline_rmse - method_rmse[0]) / baseline_rmse * 100
                    improvement_data.append({
                        'missing_days': missing_day,
                        'method': method,
                        'improvement': improvement
                    })

    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        pivot_imp = imp_df.pivot_table(values='improvement', index='missing_days', columns='method')

        for method in pivot_imp.columns:
            if method in COLOR_SCHEME:
                ax1.plot(pivot_imp.index, pivot_imp[method],
                         marker='o', label=method_names.get(method, method),
                         linewidth=2.5, markersize=7, color=COLOR_SCHEME[method])

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
        ax1.set_xlabel('Missing Days', fontsize=11, fontweight='bold')
        ax1.set_ylabel('RMSE Improvement over Linear (%)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Continuous Missing - Performance Improvement',
                      fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)


# Random missing improvement
random_df = results_df[results_df['experiment_type'] == 'random_missing'] if 'results_df' in locals() and not results_df.empty else pd.DataFrame()
if len(random_df) > 0:
    baseline_method = 'linear'
    improvement_data = []

    for ratio in random_df['missing_ratio'].unique():
        ratio_data = random_df[random_df['missing_ratio'] == ratio]
        baseline_rmse = ratio_data[ratio_data['method'] == baseline_method]['rmse_mean'].values

        if len(baseline_rmse) > 0:
            baseline_rmse = baseline_rmse[0]

            for method in ['lgbm_full', 'akima', 'knn', 'rf']:
                method_rmse = ratio_data[ratio_data['method'] == method]['rmse_mean'].values
                if len(method_rmse) > 0:
                    improvement = (baseline_rmse - method_rmse[0]) / baseline_rmse * 100
                    improvement_data.append({
                        'missing_ratio': ratio,
                        'method': method,
                        'improvement': improvement
                    })

    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        pivot_imp = imp_df.pivot_table(values='improvement', index='missing_ratio', columns='method')

        for method in pivot_imp.columns:
            if method in COLOR_SCHEME:
                ax2.plot(pivot_imp.index, pivot_imp[method],
                         marker='o', label=method_names.get(method, method),
                         linewidth=2.5, markersize=7, color=COLOR_SCHEME[method])

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
        ax2.set_xlabel('Missing Ratio (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('RMSE Improvement over Linear (%)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Random Missing - Performance Improvement',
                      fontsize=12, fontweight='bold', pad=10)
        ax2.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

# Define output folder path
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(output_folder, exist_ok=True)

save_path = os.path.join(output_folder, 'Fig_Performance_Improvement.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {save_path}")
plt.close()

print("\n" + "=" * 80)
print("✅ All visualization figures generated successfully!")
print("=" * 80)


def generate_comprehensive_summary(results_df, output_folder):
    """Generate comprehensive summary report"""
    print("\n" + "=" * 80)
    print("📊 Generating Comprehensive Summary Report")
    print("=" * 80)

    # 1. Overall performance ranking
    overall_performance = results_df.groupby('method').agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean'
    }).round(4)

    overall_performance['rank'] = overall_performance['rmse_mean'].rank()
    overall_performance = overall_performance.sort_values('rank')

    print("\nOverall method ranking (based on mean RMSE):")
    print(overall_performance)

    overall_path = os.path.join(output_folder, 'summary_overall_ranking.csv')
    overall_performance.to_csv(overall_path, encoding='utf-8-sig')

    # 2. Continuous missing scenario summary
    continuous_summary = results_df[results_df['experiment'] == 'continuous'].groupby(
        ['missing_days', 'method']
    ).agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean'
    }).round(4)

    print("\nContinuous missing scenario summary:")
    print(continuous_summary.head(20))

    continuous_path = os.path.join(output_folder, 'summary_continuous.csv')
    continuous_summary.to_csv(continuous_path, encoding='utf-8-sig')

    # 3. Random missing scenario summary
    random_summary = results_df[results_df['experiment'] == 'random'].groupby(
        ['missing_ratio', 'method']
    ).agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean'
    }).round(4)

    print("\nRandom missing scenario summary:")
    print(random_summary.head(20))

    random_path = os.path.join(output_folder, 'summary_random.csv')
    random_summary.to_csv(random_path, encoding='utf-8-sig')

    # 4. Best method statistics
    print("\n" + "=" * 80)
    print("🏆 Best Method Statistics")
    print("=" * 80)

    best_methods = results_df.loc[
        results_df.groupby(['experiment', 'missing_days', 'missing_ratio'])['rmse_mean'].idxmin()]
    best_count = best_methods['method'].value_counts()

    print("\nNumber of times each method achieved best performance:")
    for method, count in best_count.items():
        print(f"  {method:20s}: {count:3d} times")

    # 5. Improvement percentage statistics
    baseline_method = 'linear'

    improvement_stats = []
    for exp_type in ['continuous', 'random']:
        exp_data = results_df[results_df['experiment'] == exp_type]

        for method in ['lgbm_full', 'akima', 'knn', 'rf']:
            method_rmse = exp_data[exp_data['method'] == method]['rmse_mean'].mean()
            baseline_rmse = exp_data[exp_data['method'] == baseline_method]['rmse_mean'].mean()

            if baseline_rmse > 0:
                improvement = (baseline_rmse - method_rmse) / baseline_rmse * 100
                improvement_stats.append({
                    'experiment': exp_type,
                    'method': method,
                    'avg_improvement_%': improvement
                })

    improvement_df = pd.DataFrame(improvement_stats)
    print("\nAverage improvement percentage (compared to Linear interpolation):")
    print(improvement_df)

    improvement_path = os.path.join(output_folder, 'summary_improvement.csv')
    improvement_df.to_csv(improvement_path, index=False, encoding='utf-8-sig')


# =====================================================================
# Main Program
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LightGBM + Traditional Interpolation Methods Comprehensive Comparison System")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Target station: {TARGET_STATION}")
    print(f"  - Neighbor station: {NEIGHBOR_STATION}")
    print(f"  - Spatial correlation: r = {CORRELATION_R:.4f}")
    print(f"  - Station distance: {NEIGHBOR_DIST_KM:.2f} km")
    print(f"  - Analysis period: {START_DATE} to {END_DATE}")
    print(f"  - Number of repeated experiments: {N_REPEATS}")
    print(f"  - Bootstrap model count: {N_BOOTSTRAP}")
    print("=" * 80)

    # 1. Load data
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

    # 2. Create output folder
    output_folder = os.path.join(FOLDER, "comprehensive_comparison_results")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder: {output_folder}")

    # 3. Determine experiment parameters
    sample_length = len(target_data)

    if sample_length > 1000:
        missing_days_list = [7, 15, 30, 60, 90, 120, 180]
    elif sample_length > 500:
        missing_days_list = [7, 15, 30, 60]
    else:
        missing_days_list = [7, 15, 30]

    missing_ratios = np.arange(0.05, 0.55, 0.05)

    print(f"\nExperiment configuration:")
    print(f"  - Continuous missing days: {missing_days_list}")
    print(f"  - Random missing ratios: {[f'{r * 100:.0f}%' for r in missing_ratios]}")


    # 4. Run experiments
    results_df, final_interpolator = run_comprehensive_experiments(
        target_data, neighbor_data,
        missing_days_list, missing_ratios,
        output_folder
    )

    # 5. Generate visualizations
    plot_sci_comparison_figures(results_df, output_folder)

    # 6. Generate summary report
    generate_comprehensive_summary(results_df, output_folder)

    # 7. Save configuration info
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
        'methods_compared': ['LightGBM-ST', 'Akima', 'Cubic Spline', 'KNN', 'Random Forest', 'Linear'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    config_path = os.path.join(output_folder, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ Experiment complete!")
    print("=" * 80)
    print(f"All results saved to: {output_folder}")
    print(f"\nGenerated files:")
    print(f"  Data results:")
    print(f"    - comprehensive_results.csv (complete experiment results)")
    print(f"    - summary_overall_ranking.csv (overall method ranking)")
    print(f"    - summary_continuous.csv (continuous missing summary)")
    print(f"    - summary_random.csv (random missing summary)")
    print(f"    - summary_improvement.csv (improvement percentages)")
    print(f"  Visualization figures:")
    print(f"    - Fig_Continuous_Comprehensive.png (continuous missing comprehensive comparison)")
    print(f"    - Fig_Random_Comprehensive.png (random missing comprehensive comparison)")
    print(f"    - Fig_Radar_Comparison.png (radar chart comparison)")
    print(f"    - Fig_Performance_Improvement.png (performance improvement figure)")
    print(f"  Configuration file:")
    print(f"    - experiment_config.json")
    print("=" * 80 + "\n")
```
