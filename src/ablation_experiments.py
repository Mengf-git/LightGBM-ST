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
import optuna  # 新增
from optuna.samplers import TPESampler  # 新增
import json

warnings.filterwarnings("ignore")

# ----------------------
# 字体统一设置
# ----------------------
rcParams['font.family'] = ['SimHei', 'Times New Roman']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 15
rcParams['legend.fontsize'] = 11
rcParams['figure.titlesize'] = 16

# =====================================================================
# 核心配置参数
# =====================================================================
FOLDER = "D:/Grade 1/GNSS-LSTM+Attention+SG/Spatial Correlation - Machine Learning/YNYS"
TARGET_STATION = "YNYS"
NEIGHBOR_STATION = "YNLJ"
TIME_COL = "YYYYMMDD"
VALUE_COL = "U(m)"


# LightGBM超参数
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
# 超参数优化模块
# =====================================================================
import optuna
from optuna.samplers import TPESampler


def optimize_lgbm_hyperparameters(target_data, neighbor_data, output_folder, n_trials=100):
    """
    使用Optuna进行LightGBM超参数优化
    确保与实验流程的特征构建完全一致
    """
    print("\n" + "=" * 80)
    print("🔍 超参数优化（Optuna Bayesian Optimization）")
    print("=" * 80)


    # 准备数据索引（与实验保持一致）
    total_length = len(target_data)
    train_end = int(0.6 * total_length)
    val_end = int(0.8 * total_length)

    train_idx = target_data.index[:train_end]
    val_idx = target_data.index[train_end:val_end]

    # ⚠️ 关键：使用train_idx.union(val_idx)拟合Bias校正（与实验一致）
    print("\n拟合Bias校正模型...")
    feature_engineer.fit_bias_correction(
        target_data.loc[train_idx.union(val_idx)],
        neighbor_data.loc[train_idx.union(val_idx)]
    )

    # ⚠️ 关键：构建特征时使用完整target_data（无人为缺失）
    df_features = feature_engineer.create_features(
        target_data,  # 使用完整数据
        neighbor_data,
        n_lags=7,
        rolling_windows=[3, 7, 14],
        include_target_lags=True,
        use_neighbor_lag0=True
    )

    # 准备训练/验证数据（与实验一致）
    train_df = df_features.loc[train_idx.union(val_idx)].dropna(subset=['target'])

    # ⚠️ 关键：按8:2分割Train/Val
    split_point = int(0.8 * len(train_df))

    train_split = train_df.iloc[:split_point]
    val_split = train_df.iloc[split_point:]

    # 提取特征和目标
    X_train = train_split.drop(columns=['target', 'neighbor'], errors='ignore')
    y_train = train_split['target']
    X_train = X_train.fillna(X_train.mean())

    X_val = val_split.drop(columns=['target', 'neighbor'], errors='ignore')
    y_val = val_split['target']
    X_val = X_val.fillna(X_train.mean())  # 用训练集均值填充

    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  特征数量: {X_train.shape[1]}")

    def objective(trial):
        """Optuna目标函数"""
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'random_state': RANDOM_SEED,

            # 超参数搜索空间
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

    # 执行优化
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED)
    )

    print(f"\n开始优化（共{n_trials}次试验）...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 输出结果
    print("\n" + "=" * 80)
    print("✅ 优化完成!")
    print("=" * 80)
    print(f"\n最佳验证MAE: {study.best_value:.4f} mm")
    print(f"\n最优超参数:")
    for key, value in study.best_params.items():
        print(f"  {key:20s}: {value}")

    # 保存优化历史图
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0])
        axes[0].set_title('Optimization History', fontsize=14, fontweight='bold')

        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[1])
        axes[1].set_title('Hyperparameter Importances', fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_folder, 'hyperparameter_optimization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 优化历史图已保存: {save_path}")
        plt.close()
    except Exception as e:
        print(f"[警告] 可视化失败: {e}")

    return study.best_params


# =====================================================================

class SpatialCorrelationFeatureEngineering:
    """空间相关性特征工程类"""

    def __init__(self, correlation_r, neighbor_dist_km, bias_value):



    def fit_bias_correction(self, target_series, neighbor_series):
        """
        在训练集上拟合OLS Bias校正模型
        U_target = a + b * U_neighbor
        """
        valid_mask = target_series.notna() & neighbor_series.notna()

        if valid_mask.sum() < 30:
            print("    [警告] 训练样本不足,使用默认参数")
            self.ols_model = {'intercept': 0.0, 'coef': 1.0}
            return

        self.ols_model = {
            'intercept': lr.intercept_,
            'coef': lr.coef_[0]
        }

        print(f"    ✓ Bias校正: U_target = {lr.intercept_:.4f} + {lr.coef_[0]:.4f} * U_neighbor")

    def apply_bias_correction(self, neighbor_series):
        """应用Bias校正"""
        if self.ols_model is None:
            return neighbor_series.copy()

        return self.ols_model['intercept'] + self.ols_model['coef'] * neighbor_series

    def create_features(self, target_series, neighbor_series,
                        n_lags=7, rolling_windows=[3, 7, 14],
                        include_target_lags=True, use_neighbor_lag0=True):
        """
        构建完整特征矩阵

        特征类别:
        1. 时间特征: doy_sin, doy_cos, month, year
        2. 邻站滞后: nbr_lag0~lagN
        3. 邻站滚动统计: rollmean, rollstd (多窗口)
        4. 目标站历史: tgt_lag1~lag3 (仅过去值)
        5. 空间元特征: neighbor_r, neighbor_dist_km
        6. Bias校正特征: nbr_adj_lag0
        7. 缺失指标: nbr_isnan_lag0
        """
        # 创建DataFrame
        df = pd.DataFrame({
            'target': target_series,
            'neighbor': neighbor_series
        })

        # === 1. 时间特征 ===
        df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)


        # === 2. 邻站滞后特征 (lag0允许用于离线插补) ===
        for lag in range(n_lags + 1):
            if lag == 0 and not use_neighbor_lag0:
                continue
            df[f'nbr_lag{lag}'] = df['neighbor'].shift(lag)

        # === 3. 邻站滚动统计 (仅使用过去窗口) ===
        for window in rolling_windows:
            # 注意shift(1)确保不包含当前值
            df[f'nbr_rollmean_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).mean()
            df[f'nbr_rollstd_{window}'] = df['neighbor'].shift(1).rolling(
                window, min_periods=1).std().fillna(0)


        # === 4. 目标站历史滞后 (仅过去值) ===
        if include_target_lags:
            for lag in [1, 2, 3, 7]:
                df[f'tgt_lag{lag}'] = df['target'].shift(lag)


        # === 5. Bias校正特征 ===
        if self.ols_model is not None:
            nbr_adj = self.apply_bias_correction(df['neighbor'])
            for lag in range(min(3, n_lags + 1)):
                df[f'nbr_adj_lag{lag}'] = nbr_adj.shift(lag)

        # === 6. 缺失指标 ===
        df['nbr_isnan_lag0'] = df['neighbor'].isna().astype(int)
        df['n_neighbors_available'] = (~df['neighbor'].isna()).astype(int)

        return df


class LightGBMSpatialInterpolator:
    """基于LightGBM的空间相关性插值器"""

    def __init__(self, feature_engineer, lgbm_params=None):
        self.feature_engineer = feature_engineer
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.models = []  # 存储Bootstrap模型集合
        self.feature_names = None

    def _prepare_train_data(self, df_features, train_idx):
        """准备训练数据"""
        train_df = df_features.loc[train_idx]

        # 只保留target非NaN的行
        train_df = train_df.dropna(subset=['target'])

        if len(train_df) < 50:
            raise ValueError(f"训练样本不足: {len(train_df)}")

        # 填充特征中的NaN (用训练集均值)
        X = X.fillna(X.mean())

        self.feature_names = X.columns.tolist()

        return X, y

    def train_single_model(self, X_train, y_train, X_val=None, y_val=None):
        """训练单个LightGBM模型"""
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
        """获取特征重要性 (平均所有模型)"""
        if not self.models:
            return None

        # 平均所有模型的特征重要性
        importances = np.zeros(len(self.feature_names))

        for model in self.models:
            importances += model.feature_importances_

        importances /= len(self.models)




class BaselineInterpolators:
    """基线插值方法"""

    @staticmethod
    def time_only_baseline(df_features, train_idx, test_idx):
        """
        Baseline A: 仅使用时间特征
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
        Baseline B: 时间特征 + 目标站历史
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
        """线性插值基线"""
        filled = series.copy()

        # 如果是DatetimeIndex,直接使用;如果是整数索引,需要转换
        if isinstance(missing_indices, pd.DatetimeIndex):
            # 直接使用DatetimeIndex进行索引
            filled.loc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.loc[missing_indices].values
        else:
            # 整数索引的情况(原有逻辑)
            if not isinstance(missing_indices, (list, np.ndarray, pd.Index)):
                raise ValueError("missing_indices must be a list, numpy array or pandas Index")

            missing_indices = np.array(missing_indices, dtype=int)
            if len(missing_indices) == 0:
                return np.array([])

            # 确保索引在有效范围内
            if (missing_indices < 0).any() or (missing_indices >= len(series)).any():
                raise ValueError("Some indices are out of bounds")

            filled.iloc[missing_indices] = np.nan
            filled = filled.interpolate(method='linear').bfill().ffill()
            return filled.iloc[missing_indices].values

def create_continuous_missing(data, n_segments, days_per_segment, seed=None):
    """创建连续缺失段 (保持与原代码一致)"""
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

        # 计算跳跃距离（避免randint参数错误）
        if remaining > 30:
            jump = np.random.randint(5, min(30, remaining // 2))
        elif remaining > 2:
            jump_max = max(2, remaining // 2)  # 确保至少为2
            jump = np.random.randint(1, jump_max)
        else:
            jump = 1

        current_pos += jump

        if current_pos >= n_total:
            break

        remaining_missing = n_missing - len(missing_indices)
        upper_bound = min(max_segment, remaining_missing + 1)



    # 确保缺失数量准确
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
    计算评估指标 + CI覆盖率

    参数:
        true_values: Series, 真实值(完整数据)
        predicted_values: array-like, 预测值(仅缺失位置)
        pred_lower: array-like, 置信下界
        pred_upper: array-like, 置信上界
        missing_indices: DatetimeIndex 或 list, 缺失位置的索引
    """
    # 确保 missing_indices 是有效的索引
    if isinstance(missing_indices, pd.DatetimeIndex):
        valid_indices = missing_indices
    else:
        valid_indices = pd.DatetimeIndex(missing_indices)

    # 转换预测值为 Series,使用 missing_indices 作为索引
    if isinstance(predicted_values, pd.Series):
        pred_series = predicted_values
    else:
        pred_series = pd.Series(predicted_values, index=valid_indices)

    if pred_lower is not None and not isinstance(pred_lower, pd.Series):
        pred_lower = pd.Series(pred_lower, index=valid_indices)

    if pred_upper is not None and not isinstance(pred_upper, pd.Series):
        pred_upper = pd.Series(pred_upper, index=valid_indices)

    # 提取真实值和预测值
    true_missing = true_values.loc[valid_indices]
    pred_missing = pred_series.loc[valid_indices]

    # 筛选有效值
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
    运行完整的LightGBM+空间相关性实验
    """
    print("\n" + "=" * 80)
    print("🚀 LightGBM + 空间相关性融合插值实验")
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
    print(f"  Train: {len(train_idx)} 天 ({train_idx[0].date()} ~ {train_idx[-1].date()})")
    print(f"  Val:   {len(val_idx)} 天 ({val_idx[0].date()} ~ {val_idx[-1].date()})")
    print(f"  Test:  {len(test_idx)} 天 ({test_idx[0].date()} ~ {test_idx[-1].date()})")

    # 在训练集上拟合Bias校正
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
            'lgbm_full': {'rmse': [], 'mae': [], 'corr': []},
            'lgbm_no_spatial': {'rmse': [], 'mae': [], 'corr': []},
            'time_only': {'rmse': [], 'mae': [], 'corr': []},
            'linear': {'rmse': [], 'mae': [], 'corr': []}
        }

        for repeat in range(N_REPEATS):
            print(f"\n  重复 {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat
            # 转换为全局索引
            missing_indices_global = test_idx[missing_indices_local]

            # 构建完整数据 (Train+Val无缺失, Test有缺失)
            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # === 构建特征 ===
            df_features = feature_engineer.create_features(
                full_target, neighbor_data,
                n_lags=7, rolling_windows=[3, 7, 14],
                include_target_lags=True, use_neighbor_lag0=True
            )

            # === 模型1: 完整LightGBM (含空间特征) ===
            interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)
            interpolator.train_ensemble_residual(df_features, train_idx.union(val_idx))
            pred_median, pred_lower, pred_upper = interpolator.predict_with_uncertainty_residual(
                df_features, missing_indices_global
            )
            metrics_time = calculate_metrics(target_data, pred_time, missing_indices_global)

            method_results['lgbm_full']['rmse'].append(metrics['rmse'])
            method_results['lgbm_full']['mae'].append(metrics['mae'])
            method_results['lgbm_full']['corr'].append(metrics['correlation'])


            # === 模型2: LightGBM (无空间特征 - 消融) ===
            # 移除空间相关性特征
            df_no_spatial = df_features.drop(columns=[
                                                         'neighbor_r', 'neighbor_dist_km', 'spatial_weight'
                                                     ] + [c for c in df_features.columns if 'nbr_adj' in c],
                                             errors='ignore')

            feature_engineer_nospatial = SpatialCorrelationFeatureEngineering(0, 0, 0)
            interpolator_nospatial = LightGBMSpatialInterpolator(feature_engineer_nospatial, LGBM_PARAMS)

            # 手动准备训练数据
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
            'lgbm_full': {'rmse': [], 'mae': [], 'corr': []},
            'lgbm_no_spatial': {'rmse': [], 'mae': [], 'corr': []},
            'time_only': {'rmse': [], 'mae': [], 'corr': []},
            'linear': {'rmse': [], 'mae': [], 'corr': []}
        }

        for repeat in range(N_REPEATS):
            print(f"\n  重复 {repeat + 1}/{N_REPEATS}")

            seed = RANDOM_SEED + repeat

            # 创建缺失 (仅在Test段)
            test_data = target_data.loc[test_idx].copy()

            masked_test, missing_indices_local = create_random_clustered_missing(
                test_data, missing_ratio, min_segment=5, max_segment=30, seed=seed
            )

            missing_indices_global = test_idx[missing_indices_local]

            # 构建完整数据
            full_target = target_data.copy()
            full_target.loc[missing_indices_global] = np.nan

            # 构建特征
            df_features = feature_engineer.create_features(
                full_target, neighbor_data,
                n_lags=7, rolling_windows=[3, 7, 14],
                include_target_lags=True, use_neighbor_lag0=True
            )

            # === 完整模型 ===
            interpolator = LightGBMSpatialInterpolator(feature_engineer, LGBM_PARAMS)

            method_results['lgbm_full']['rmse'].append(metrics['rmse'])
            method_results['lgbm_full']['mae'].append(metrics['mae'])
            method_results['lgbm_full']['corr'].append(metrics['correlation'])
            method_results['lgbm_full']['ci_cov'].append(metrics['ci_coverage'])

            # === 无空间特征模型 ===
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
    results_path = os.path.join(output_folder, 'lgbm_spatial_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: {results_path}")

    return results_df, interpolator


def plot_feature_importance(interpolator, output_folder):
    """绘制特征重要性"""
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
    print(f"✓ 特征重要性图已保存: {save_path}")
    plt.close()


def plot_comparison_results(results_df, output_folder):
    """绘制方法对比图"""

    # 图1: 连续缺失 - RMSE对比
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
        print(f"✓ 连续缺失对比图已保存: {save_path}")
        plt.close()

    # 图2: 随机缺失 - RMSE对比
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
        print(f"✓ 随机缺失对比图已保存: {save_path}")
        plt.close()


def generate_summary_report(results_df, output_folder):
    """生成汇总报告"""
    print("\n" + "=" * 80)
    print("📊 生成汇总报告")
    print("=" * 80)

    # 1. 总体性能
    summary_overall = results_df.groupby('method').agg({
        'rmse_mean': 'mean',
        'mae_mean': 'mean',
        'correlation_mean': 'mean',
        'ci_coverage_mean': 'mean'
    }).round(4)

    print("\n总体平均性能:")
    print(summary_overall)

    overall_path = os.path.join(output_folder, 'summary_overall.csv')
    summary_overall.to_csv(overall_path, encoding='utf-8-sig')

    # 2. 按实验类型汇总
    summary_by_exp = results_df.groupby(['experiment', 'method']).agg({
        'rmse_mean': ['mean', 'std'],
        'mae_mean': ['mean', 'std'],
        'correlation_mean': ['mean', 'std']
    }).round(4)

    print("\n按实验类型汇总:")
    print(summary_by_exp)

    exp_path = os.path.join(output_folder, 'summary_by_experiment.csv')
    summary_by_exp.to_csv(exp_path, encoding='utf-8-sig')

    # 3. CI覆盖率分析
    ci_coverage = results_df[results_df['method'] == 'lgbm_full']['ci_coverage_mean'].dropna()

    if len(ci_coverage) > 0:
        print(f"\n95% CI覆盖率统计 (LGBM_Full):")
        print(f"  平均: {ci_coverage.mean():.3f}")
        print(f"  标准差: {ci_coverage.std():.3f}")
        print(f"  中位数: {ci_coverage.median():.3f}")
        print(f"  范围: [{ci_coverage.min():.3f}, {ci_coverage.max():.3f}]")

    # 4. 方法排名
    print("\n" + "=" * 80)
    print("🏆 方法排名 (基于平均RMSE)")
    print("=" * 80)

    ranking = results_df.groupby('method')['rmse_mean'].mean().sort_values()
    for rank, (method, rmse) in enumerate(ranking.items(), 1):
        print(f"  {rank}. {method:20s} - RMSE: {rmse:.4f} mm")

    # 5. 改进百分比
    if 'lgbm_full' in ranking.index and 'linear' in ranking.index:
        improvement = (ranking['linear'] - ranking['lgbm_full']) / ranking['linear'] * 100
        print(f"\n✨ LGBM_Full 相比 Linear 改进: {improvement:.2f}%")

    if 'lgbm_full' in ranking.index and 'lgbm_no_spatial' in ranking.index:
        spatial_benefit = (ranking['lgbm_no_spatial'] - ranking['lgbm_full']) / ranking['lgbm_no_spatial'] * 100
        print(f"✨ 空间特征贡献: {spatial_benefit:.2f}%")


# =====================================================================
# 主程序
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LightGBM + 空间相关性融合插值系统")
    print("=" * 80)
    print(f"配置信息:")
    print(f"  - 目标站点: {TARGET_STATION}")
    print(f"  - 邻近站点: {NEIGHBOR_STATION}")
    print(f"  - 空间相关性: r = {CORRELATION_R:.4f}")
    print(f"  - 站点距离: {NEIGHBOR_DIST_KM:.2f} km")
    print(f"  - 分析时间段: {START_DATE} 至 {END_DATE}")
    print(f"  - 重复实验次数: {N_REPEATS}")
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

        # 处理数据
        df_target = df_target.set_index(TIME_COL).sort_index()
        df_neighbor = df_neighbor.set_index(TIME_COL).sort_index()

        # 提取指定时间段
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)

        target_data = df_target.loc[start_dt:end_dt, VALUE_COL]
        neighbor_data = df_neighbor.loc[start_dt:end_dt, VALUE_COL]

        # 重新索引确保连续
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
    output_folder = os.path.join(FOLDER, "lgbm_spatial_results")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ 输出文件夹: {output_folder}")
    # ===================================================================
    # 🔥 超参数优化（可选）
    # ===================================================================
    ENABLE_HYPERPARAMETER_OPTIMIZATION = True  # 首次运行设为True，后续设为False

    if ENABLE_HYPERPARAMETER_OPTIMIZATION:
        print("\n" + "=" * 80)
        print("⚙️ 开始超参数优化（约需10-20分钟）")
        print("=" * 80)

        best_params = optimize_lgbm_hyperparameters(
            target_data, neighbor_data, output_folder,
            n_trials=100  # 可调整：50(快速), 100(推荐), 200(精细)
        )

        # 更新全局参数
        LGBM_PARAMS.update(best_params)

        # 保存最优参数
        best_params_path = os.path.join(output_folder, 'best_hyperparameters.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4, ensure_ascii=False)

        print(f"\n✓ 最优超参数已保存: {best_params_path}")
        print("\n💡 提示: 后续运行可将 ENABLE_HYPERPARAMETER_OPTIMIZATION 设为 False")
        print("         并手动更新 LGBM_PARAMS 字典以节省时间")
    else:
        print("\n⏭️ 跳过超参数优化，使用预设参数")

    # ===================================================================
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

    # 4. 运行实验
    results_df, final_interpolator = run_spatial_lgbm_experiments(
        target_data, neighbor_data,
        missing_days_list, missing_ratios,
        output_folder
    )

    # 5. 生成可视化
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)

    plot_feature_importance(final_interpolator, output_folder)
    plot_comparison_results(results_df, output_folder)

    # 6. 生成汇总报告
    generate_summary_report(results_df, output_folder)

    # 7. 保存配置信息
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
    print("✅ 实验完成!")
    print("=" * 80)
    print(f"所有结果已保存至: {output_folder}")
    print(f"  - 详细结果: lgbm_spatial_results.csv")
    print(f"  - 总体汇总: summary_overall.csv")
    print(f"  - 实验汇总: summary_by_experiment.csv")
    print(f"  - 特征重要性图: feature_importance.png")
    print(f"  - 连续缺失对比: continuous_comparison.png")
    print(f"  - 随机缺失对比: random_comparison.png")
    print(f"  - 实验配置: experiment_config.json")
    print("=" * 80 + "\n")
