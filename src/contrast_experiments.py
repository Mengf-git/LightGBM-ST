"""
GNSS时间序列缺失值插补对比实验
===========================================================
1. 集成Akima、KNN、随机森林、三次样条插值方法
2. 生成符合SCI论文标准的可视化对比图，可视化缺失和插值
3. 完整的方法性能对比实验
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

# 设置颜色方案
COLOR_SCHEME = {
    'lgbm_full': '#1f77b4',  # 蓝色
    'akima': '#ff7f0e',  # 橙色
    'cubic_spline': '#2ca02c',  # 绿色
    'knn': '#d62728',  # 红色
    'rf': '#9467bd',  # 紫色
    'lgbm_no_spatial': '#8c564b',  # 棕色
    'time_only': '#e377c2',  # 粉色
    'linear': '#7f7f7f'  # 灰色
}

# =====================================================================
# 核心配置参数
# =====================================================================
FOLDER = "../data/real_GNSS"
TARGET_STATION = "YNYS"
NEIGHBOR_STATION = "YNLJ"
TIME_COL = "YYYYMMDD"
VALUE_COL = "U(m)"
START_DATE = "2011-12-05"
END_DATE = "2019-02-17"
N_REPEATS = 10
N_BOOTSTRAP = 80
RANDOM_SEED = 42

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
# 传统插值方法类
# =====================================================================
class TraditionalInterpolationMethods:
    """集成传统插值方法"""

    @staticmethod
    def cubic_spline_interpolation(series):
        """三次样条插值"""
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
            print(f"    [警告] 三次样条插值失败: {e}")
            return series.interpolate(method='linear').bfill().ffill()

    @staticmethod
    def akima_interpolation(series):
        """Akima插值"""
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
            print(f"    [警告] Akima插值失败: {e}")
            return TraditionalInterpolationMethods.cubic_spline_interpolation(series)

    @staticmethod
    def _create_lagged_features(series, n_lags=30, n_rolling_stats=3):
        """为时间序列创建滞后特征和滚动统计特征"""
        n = len(series)
        features = np.zeros((n, n_lags + n_rolling_stats * 4))

        series_filled = series.fillna(method='ffill').fillna(method='bfill')

        # 滞后特征
        for lag in range(n_lags):
            if lag == 0:
                features[:, lag] = series_filled.values
            else:
                features[lag:, lag] = series_filled.iloc[:-lag].values
                features[:lag, lag] = series_filled.iloc[0]

        # 滚动统计特征
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
        """KNN插值"""
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
            print(f"    [警告] KNN插值失败: {e}")
            return series.interpolate(method='linear').bfill().ffill()

    @staticmethod
    def random_forest_interpolation(series, n_estimators=100, max_depth=10,
                                    min_samples_leaf=2, n_iterations=10):
        """随机森林迭代插值"""
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
            print(f"    [警告] 随机森林插值失败: {e}")
            return series.interpolate(method='linear').bfill().ffill()


# =====================================================================

class SpatialCorrelationFeatureEngineering:
    """空间相关性特征工程类"""

    def __init__(self, correlation_r, neighbor_dist_km, bias_value):
        self.bias_value = bias_value
        self.ols_model = None
        self.base_model = None
        self.residuals = None
        self.residual_std = None

    def fit_bias_correction(self, target_series, neighbor_series):
        valid_mask = target_series.notna() & neighbor_series.notna()

        if valid_mask.sum() < 30:
            print("    [警告] 训练样本不足,使用默认参数")
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

        print(f"    ✓ Bias校正: U_target = {lr.intercept_:.4f} + {lr.coef_[0]:.4f} * U_neighbor")

    def apply_bias_correction(self, neighbor_series):
        if self.ols_model is None:
            return neighbor_series.copy()

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
    """基于LightGBM的空间相关性插值器"""

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
            raise ValueError(f"训练样本不足: {len(train_df)}")

        X = train_df.drop(columns=['target', 'neighbor'], errors='ignore')
        y = train_df['target']
        X = X.fillna(X.mean())

        self.feature_names = X.columns.tolist()

        return X, y

    def train_ensemble_residual(self, df_features, train_idx, n_bootstrap=50):
        print(f"    训练残差Bootstrap集成...")

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

        print(f"    ✓ 基础模型训练完成")
        print(f"    ✓ 训练集残差标准差: {residual_std:.4f} mm")
        print(f"    ✓ 验证集MAE: {val_mae:.4f} mm")

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
    """基线插值方法"""

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
    """创建连续缺失段"""
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
    """创建随机聚集缺失"""
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
    """计算评估指标 + CI覆盖率"""
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
    运行完整的对比实验（包含传统插值方法）
    """
    print("\n" + "=" * 80)
    print("🚀 LightGBM + 传统插值方法综合对比实验")
    print("=" * 80)

    # 初始化特征工程
    feature_engineer = SpatialCorrelationFeatureEngineering(
        correlation_r=CORRELATION_R,
        neighbor_dist_km=NEIGHBOR_DIST_KM,
        bias_value=BIAS_VALUE
    )

    # 准备数据索引
    total_length = len(target_data)
    train_end = int(0.6 * total_length)
    val_end = int(0.8 * total_length)

    train_idx = target_data.index[:train_end]
    val_idx = target_data.index[train_end:val_end]
    test_idx = target_data.index[val_end:]

    print(f"\n数据划分:")
    print(f"  Train: {len(train_idx)} 天")
    print(f"  Val:   {len(val_idx)} 天")
    print(f"  Test:  {len(test_idx)} 天")

    # 拟合Bias校正
    print("\n拟合Bias校正模型...")
    feature_engineer.fit_bias_correction(
        target_data.loc[train_idx.union(val_idx)],
        neighbor_data.loc[train_idx.union(val_idx)]
    )

    results_list = []

    # ====================================================================
    # 实验1: 连续缺失
    # ====================================================================
    print("\n" + "=" * 80)
    print("🔴 实验1: 连续缺失段")
    print("=" * 80)

    for missing_days in missing_days_list:
        print(f"\n{'─' * 80}")
        print(f"缺失段长度: {missing_days} 天")
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
            print(f"\n  重复 {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat

            # 创建缺失
            test_data = target_data.loc[test_idx].copy()
            n_segments = max(1, int(len(test_idx) * 0.1 / missing_days))

            masked_test, missing_indices_local, _ = create_continuous_missing(
                test_data, n_segments, missing_days, seed=seed
            )

            missing_indices_global = test_idx[missing_indices_local]

            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # === LightGBM-ST方法 ===
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

            # === Akima插值 ===
            pred_akima = TraditionalInterpolationMethods.akima_interpolation(full_target)
            metrics_akima = calculate_metrics_with_ci(
                target_data, pred_akima, None, None, missing_indices_global
            )
            method_results['akima']['rmse'].append(metrics_akima['rmse'])
            method_results['akima']['mae'].append(metrics_akima['mae'])
            method_results['akima']['corr'].append(metrics_akima['correlation'])

            # === 三次样条插值 ===
            pred_cubic = TraditionalInterpolationMethods.cubic_spline_interpolation(full_target)
            metrics_cubic = calculate_metrics_with_ci(
                target_data, pred_cubic, None, None, missing_indices_global
            )
            method_results['cubic_spline']['rmse'].append(metrics_cubic['rmse'])
            method_results['cubic_spline']['mae'].append(metrics_cubic['mae'])
            method_results['cubic_spline']['corr'].append(metrics_cubic['correlation'])

            # === KNN插值 ===
            pred_knn = TraditionalInterpolationMethods.knn_interpolation(full_target)
            metrics_knn = calculate_metrics_with_ci(
                target_data, pred_knn, None, None, missing_indices_global
            )
            method_results['knn']['rmse'].append(metrics_knn['rmse'])
            method_results['knn']['mae'].append(metrics_knn['mae'])
            method_results['knn']['corr'].append(metrics_knn['correlation'])

            # === 随机森林插值 ===
            pred_rf = TraditionalInterpolationMethods.random_forest_interpolation(full_target)
            metrics_rf = calculate_metrics_with_ci(
                target_data, pred_rf, None, None, missing_indices_global
            )
            method_results['rf']['rmse'].append(metrics_rf['rmse'])
            method_results['rf']['mae'].append(metrics_rf['mae'])
            method_results['rf']['corr'].append(metrics_rf['correlation'])

            # === 线性插值 ===
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

        # 汇总结果
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
    # 实验2: 随机聚集缺失
    # ====================================================================
    print("\n" + "=" * 80)
    print("🔵 实验2: 随机聚集缺失")
    print("=" * 80)

    for missing_ratio in missing_ratios:
        print(f"\n{'─' * 80}")
        print(f"缺失比例: {missing_ratio * 100:.1f}%")
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
            print(f"\n  重复 {repeat + 1}/{N_REPEATS}")

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

            # === 传统方法 ===
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

        # 汇总结果
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

    # 保存结果
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(output_folder, 'comprehensive_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: {results_path}")

    return results_df, interpolator


def plot_interpolation_comparison_figures(target_data, neighbor_data, feature_engineer,
                                          missing_days_list, missing_ratios, output_folder):
    """
    生成真实值vs预测值的插补效果对比图
    专注于展示各方法的实际插补效果
    """
    print("\n" + "=" * 80)
    print("📊 生成插补效果可视化对比图")
    print("=" * 80)

    # 定义方法显示名称
    method_names = {
        'lgbm_full': 'LightGBM-ST',
        'akima': 'Akima',
        'cubic_spline': 'Cubic Spline',
        'knn': 'KNN',
        'rf': 'Random Forest',
        'linear': 'Linear'
    }

    # 准备数据索引
    total_length = len(target_data)
    train_end = int(0.6 * total_length)
    val_end = int(0.8 * total_length)
    train_idx = target_data.index[:train_end]
    val_idx = target_data.index[train_end:val_end]
    test_idx = target_data.index[val_end:]

    # ====================================================================
    # 图1: 连续缺失场景 - 不同缺失长度的插补效果对比
    # ====================================================================
    print("\n生成连续缺失场景插补效果图...")

    # 选择代表性的缺失天数
    selected_days = [7, 30, 90] if 90 in missing_days_list else [7, 30]

    for missing_days in selected_days:
        print(f"  处理 {missing_days} 天缺失场景...")

        # 创建缺失数据（使用固定种子以保证可重复性）
        test_data = target_data.loc[test_idx].copy()
        n_segments = 1  # 只创建一个缺失段用于可视化

        masked_test, missing_indices_local, _ = create_continuous_missing(
            test_data, n_segments, missing_days, seed=RANDOM_SEED
        )

        missing_indices_global = test_idx[missing_indices_local]

        # 构建完整数据
        full_target = target_data.copy()
        full_target.loc[missing_indices_global] = np.nan

        # 准备特征
        df_features = feature_engineer.create_features(
            full_target, neighbor_data,
            n_lags=7, rolling_windows=[3, 7, 14],
            include_target_lags=True, use_neighbor_lag0=False
        )

        # === 执行各种插值方法 ===
        predictions = {}

        # LightGBM-ST
        interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
        interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx), n_bootstrap=30)
        pred_lgbm, _, _ = interpolator.predict_with_uncertainty_residual(df_features, missing_indices_global)
        predictions['lgbm_full'] = pred_lgbm

        # Akima插值
        pred_akima = TraditionalInterpolationMethods.akima_interpolation(full_target)
        predictions['akima'] = pred_akima.loc[missing_indices_global].values

        # 三次样条插值
        pred_cubic = TraditionalInterpolationMethods.cubic_spline_interpolation(full_target)
        predictions['cubic_spline'] = pred_cubic.loc[missing_indices_global].values

        # KNN插值
        pred_knn = TraditionalInterpolationMethods.knn_interpolation(full_target)
        predictions['knn'] = pred_knn.loc[missing_indices_global].values

        # 随机森林插值
        pred_rf = TraditionalInterpolationMethods.random_forest_interpolation(full_target)
        predictions['rf'] = pred_rf.loc[missing_indices_global].values

        # 线性插值
        pred_linear = BaselineInterpolators.linear_interpolation(full_target, missing_indices_global)
        predictions['linear'] = pred_linear

        # === 绘制对比图 ===
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25)

        # 获取真实值
        true_values = target_data.loc[missing_indices_global].values
        x_axis = np.arange(len(missing_indices_global))

        # 扩展窗口：显示缺失段前后各10天的数据
        window_before = 10
        window_after = 10

        start_idx_local = missing_indices_local[0]
        end_idx_local = missing_indices_local[-1]

        extended_start = max(0, start_idx_local - window_before)
        extended_end = min(len(test_data) - 1, end_idx_local + window_after)

        extended_indices = test_idx[extended_start:extended_end + 1]
        extended_true = target_data.loc[extended_indices].values
        extended_x = np.arange(len(extended_indices))

        # 缺失段在扩展窗口中的位置
        missing_start_in_window = start_idx_local - extended_start
        missing_end_in_window = end_idx_local - extended_start + 1
        missing_x_in_window = np.arange(missing_start_in_window, missing_end_in_window)

        # 绘制6个子图
        methods_to_plot = ['lgbm_full', 'akima', 'cubic_spline', 'knn', 'rf', 'linear']

        for idx, method in enumerate(methods_to_plot):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            # 绘制完整的观测数据（浅色）
            ax.plot(extended_x, extended_true, 'o-', color='#CCCCCC',
                    linewidth=1.5, markersize=4, alpha=0.6, label='Observed Data')

            # 高亮显示缺失段的真实值
            ax.plot(missing_x_in_window, true_values, 'o', color='#000000',
                    markersize=6, markeredgewidth=1.5, markerfacecolor='white',
                    label='True Values (Missing)', zorder=5)

            # 绘制预测值
            pred_values = predictions[method]
            ax.plot(missing_x_in_window, pred_values, 's-',
                    color=COLOR_SCHEME.get(method, '#1f77b4'),
                    linewidth=2.5, markersize=7, label=f'{method_names[method]} Prediction',
                    zorder=4)

            # 标注缺失区域
            ax.axvspan(missing_start_in_window, missing_end_in_window - 1,
                       alpha=0.15, color='red', label='Missing Period')

            # 计算并显示RMSE和R
            rmse = np.sqrt(mean_squared_error(true_values, pred_values))
            try:
                corr, _ = pearsonr(true_values, pred_values)
            except:
                corr = 0.0

            # 设置标题和标签
            ax.set_title(f'{method_names[method]}\nRMSE={rmse:.3f} mm, R={corr:.3f}',
                         fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel('Time (days)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Displacement (mm)', fontsize=10, fontweight='bold')

            # 图例
            ax.legend(loc='upper left', fontsize=8, frameon=True,
                      fancybox=False, edgecolor='black')

            # 网格和边框
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 总标题
        fig.suptitle(f'Interpolation Performance Comparison - {missing_days}-day Continuous Missing',
                     fontsize=14, fontweight='bold', y=0.995)

        save_path = os.path.join(output_folder, f'Fig_Interpolation_Continuous_{missing_days}days.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ 已保存: {save_path}")
        plt.close()

    # ====================================================================
    # 图2: 随机聚集缺失场景 - 插补效果对比
    # ====================================================================
    print("\n生成随机聚集缺失场景插补效果图...")

    # 选择代表性的缺失比例
    selected_ratios = [0.10, 0.20, 0.30] if 0.30 in missing_ratios else [0.10, 0.20]

    for missing_ratio in selected_ratios:
        print(f"  处理 {missing_ratio * 100:.0f}% 缺失比例场景...")

        # 创建缺失数据
        test_data = target_data.loc[test_idx].copy()

        masked_test, missing_indices_local = create_random_clustered_missing(
            test_data, missing_ratio, min_segment=5, max_segment=30, seed=RANDOM_SEED
        )

        missing_indices_global = test_idx[missing_indices_local]

        # 构建完整数据
        full_target = target_data.copy()
        full_target.loc[missing_indices_global] = np.nan

        # 准备特征
        df_features = feature_engineer.create_features(
            full_target, neighbor_data,
            n_lags=7, rolling_windows=[3, 7, 14],
            include_target_lags=True, use_neighbor_lag0=True
        )

        # === 执行各种插值方法 ===
        predictions = {}

        # LightGBM-ST
        interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
        interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx), n_bootstrap=30)
        pred_lgbm, _, _ = interpolator.predict_with_uncertainty_residual(df_features, missing_indices_global)
        predictions['lgbm_full'] = pred_lgbm

        # 其他方法
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


# 随机缺失改进
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

# 定义输出文件夹路径
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(output_folder, exist_ok=True)

save_path = os.path.join(output_folder, 'Fig_Performance_Improvement.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ 已保存: {save_path}")
plt.close()

print("\n" + "=" * 80)
print("✅ 所有可视化图表生成完成!")
print("=" * 80)


def generate_comprehensive_summary(results_df, output_folder):
    """生成综合汇总报告"""
    print("\n" + "=" * 80)
    print("📊 生成综合汇总报告")
    print("=" * 80)

    # 1. 总体性能排名
    overall_performance = results_df.groupby('method').agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean'
    }).round(4)

    overall_performance['rank'] = overall_performance['rmse_mean'].rank()
    overall_performance = overall_performance.sort_values('rank')

    print("\n总体方法排名 (基于平均RMSE):")
    print(overall_performance)

    overall_path = os.path.join(output_folder, 'summary_overall_ranking.csv')
    overall_performance.to_csv(overall_path, encoding='utf-8-sig')

    # 2. 连续缺失场景汇总
    continuous_summary = results_df[results_df['experiment'] == 'continuous'].groupby(
        ['missing_days', 'method']
    ).agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean'
    }).round(4)

    print("\n连续缺失场景汇总:")
    print(continuous_summary.head(20))

    continuous_path = os.path.join(output_folder, 'summary_continuous.csv')
    continuous_summary.to_csv(continuous_path, encoding='utf-8-sig')

    # 3. 随机缺失场景汇总
    random_summary = results_df[results_df['experiment'] == 'random'].groupby(
        ['missing_ratio', 'method']
    ).agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean'
    }).round(4)

    print("\n随机缺失场景汇总:")
    print(random_summary.head(20))

    random_path = os.path.join(output_folder, 'summary_random.csv')
    random_summary.to_csv(random_path, encoding='utf-8-sig')

    # 4. 最佳方法统计
    print("\n" + "=" * 80)
    print("🏆 最佳方法统计")
    print("=" * 80)

    best_methods = results_df.loc[
        results_df.groupby(['experiment', 'missing_days', 'missing_ratio'])['rmse_mean'].idxmin()]
    best_count = best_methods['method'].value_counts()

    print("\n各方法获得最佳性能的次数:")
    for method, count in best_count.items():
        print(f"  {method:20s}: {count:3d} 次")

    # 5. 改进百分比统计
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
    print("\n平均改进百分比 (相比Linear插值):")
    print(improvement_df)

    improvement_path = os.path.join(output_folder, 'summary_improvement.csv')
    improvement_df.to_csv(improvement_path, index=False, encoding='utf-8-sig')


# =====================================================================
# 主程序
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LightGBM + 传统插值方法综合对比系统")
    print("=" * 80)
    print(f"配置信息:")
    print(f"  - 目标站点: {TARGET_STATION}")
    print(f"  - 邻近站点: {NEIGHBOR_STATION}")
    print(f"  - 分析时间段: {START_DATE} 至 {END_DATE}")
    print(f"  - 重复实验次数: {N_REPEATS}")
    print(f"  - Bootstrap模型数: {N_BOOTSTRAP}")
    print("=" * 80)

    # 1. 加载数据
    print("\n加载数据...")
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

        print(f"✓ 数据加载成功")
        print(f"  - 目标站: {len(target_data)} 天, 缺失 {target_data.isna().sum()} 天")
        print(f"  - 邻站: {len(neighbor_data)} 天, 缺失 {neighbor_data.isna().sum()} 天")

    except Exception as e:
        print(f"[错误] 数据加载失败: {e}")
        exit(1)

    # 2. 创建输出文件夹
    output_folder = os.path.join(FOLDER, "comprehensive_comparison_results")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ 输出文件夹: {output_folder}")

    # 3. 确定实验参数
    sample_length = len(target_data)

    if sample_length > 1000:
        missing_days_list = [7, 15, 30, 60, 90, 120, 180]
    elif sample_length > 500:
        missing_days_list = [7, 15, 30, 60]
    else:
        missing_days_list = [7, 15, 30]

    missing_ratios = np.arange(0.05, 0.55, 0.05)

    print(f"\n实验配置:")
    print(f"  - 连续缺失天数: {missing_days_list}")
    print(f"  - 随机缺失比例: {[f'{r * 100:.0f}%' for r in missing_ratios]}")

    # 导入所需函数
    from LightGBM和其他方法对比 import plot_sci_comparison_figures, generate_comprehensive_summary

    # 4. 运行实验
    results_df, final_interpolator = run_comprehensive_experiments(
        target_data, neighbor_data,
        missing_days_list, missing_ratios,
        output_folder
    )

    # 5. 生成可视化
    plot_sci_comparison_figures(results_df, output_folder)

    # 6. 生成汇总报告
    generate_comprehensive_summary(results_df, output_folder)

    # 7. 保存配置信息
    config_info = {
        'target_station': TARGET_STATION,
        'neighbor_station': NEIGHBOR_STATION,
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
    print("✅ 实验完成!")
    print("=" * 80)
    print(f"所有结果已保存至: {output_folder}")
    print(f"\n生成的文件:")
    print(f"  数据结果:")
    print(f"    - comprehensive_results.csv (完整实验结果)")
    print(f"    - summary_continuous.csv (连续缺失汇总)")
    print(f"    - summary_random.csv (随机缺失汇总)")
    print(f"    - summary_improvement.csv (改进百分比)")
    print(f"  可视化图表:")
    print(f"    - Fig_Continuous_Comprehensive.png (连续缺失综合对比)")
    print(f"    - Fig_Random_Comprehensive.png (随机缺失综合对比)")
    print(f"  配置文件:")
    print(f"    - experiment_config.json")
    print("=" * 80 + "\n")
