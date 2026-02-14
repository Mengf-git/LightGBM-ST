"""
核心功能：
1. 时间对齐与共同样本数统计
2. Haversine距离计算
3. Pearson/Spearman相关系数（含bootstrap CI和permutation test）
4. 幅值差异指标（Bias, RMSE, MAE, σ比）
5. 互相关与滞后分析
6. 输出完整的pairwise_metrics.csv与可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保可复现
np.random.seed(0)


class GNSSCorrelationAnalysis:
    """GNSS站点空间相关性分析类"""

    def __init__(self, nboot=2000, nperm=2000):
        """
        参数:
            nboot: Bootstrap重采样次数
            nperm: Permutation置换检验次数
        """
        self.nboot = nboot
        self.nperm = nperm

    def haversine_distance(self, lat1, lon1, lat2, lon2):

        R = 6371.0  # 地球半径（km）

        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def bootstrap_correlation(self, x, y):
        """
        Bootstrap方法计算Pearson相关系数的95%置信区间

        返回: (r均值, CI下界, CI上界)
        """
        n = len(x)
        r_boot = np.zeros(self.nboot)

        for i in range(self.nboot):
            idx = np.random.choice(n, size=n, replace=True)

        r_mean = np.mean(r)
        ci_lower = np.percentile(r_boot, 2.5)
        ci_upper = np.percentile(r_boot, 97.5)

        return r_mean, ci_lower, ci_upper

    def permutation_test(self, x, y):
        """
        Permutation置换检验计算p-value

        H0: x和y之间无相关性
        """
        r_obs = stats.pearsonr(x, y)[0]
        r_perm = np.zeros(self.nperm)

        for i in range(self.nperm):
            r_perm[i] = stats.pearsonr(x, y_shuffled)[0]

        p_perm = np.mean(np.abs(r_perm) >= np.abs(r_obs))

        return p_perm

    def cross_correlation(self, x, y, max_lag=180):
        """
        计算互相关并找到最大相关及对应滞后

        参数:
            max_lag: 最大滞后天数（建议≤N/2）

        返回: (最大互相关值, 对应滞后天数)
        """
        # 标准化序列
        x_norm = (x - np.mean(x)) / (np.std(x) * len(x))
        y_norm = (y - np.mean(y)) / np.std(y)

        # 计算互相关
        correlation = signal.correlate(y_norm, x_norm, mode='full')
        lags = signal.correlation_lags(len(x), len(y), mode='full')

        # 限制在max_lag范围内
        mask = np.abs(lags) - max_lag
        correlation = correlation[mask]
        lags = lags[mask]

        # 找到最大值
        max_idx = np.argmax(np.abs(correlation))
        cc_max = correlation[max_idx]
        lag_at_max = lags[max_idx]

        return cc_max, lag_at_max

    def detrend_series(self, series, method='linear'):
        """
        去趋势处理

        参数:
            method: 'linear' 或 'constant'
        """
        if method == 'linear':
            return signal.detrend(series, type='linear')
        else:
            return series - np.mean(series)

    def analyze_pair(self, df1, df2, station1_name, station2_name,
                     lat1, lon1, lat2, lon2,
                     date_col='YYYYMMDD', value_col='U(m)',
                     detrend=False):

        # 1. 时间对齐（取交集）
        df1[date_col] = pd.to_datetime(df1[date_col], format='%Y%m%d')
        df2[date_col] = pd.to_datetime(df2[date_col], format='%Y%m%d')

        merged = pd.merge(df1[[date_col, value_col]],
                          df2[[date_col, value_col]],
                          suffixes=('_1', '_2'))

        # 去除NaN
        merged = merged.dropna()
        N = len(merged)

        if N < 30:
            print(f"警告: {station1_name}-{station2_name} 共同样本数仅{N}，结果可能不可靠")

        x = merged[f'{value_col}_1'].values
        y = merged[f'{value_col}_2'].values

        # 去趋势（可选）
        if detrend:
            x = self.detrend_series(x)
            y = self.detrend_series(y)

        # 2. Haversine距离
        distance_km = self.haversine(lat1, lon1, lat2, lon2)

        # 3. Pearson相关（含经典p值、bootstrap CI、permutation p）
        r_mean, r_ci_lo, r_ci_hi = self.bootstrap_correlation(x, y)
        p_perm = self.permutation_test(x, y)

        # 4. 幅值差异指标
        bias = np.mean(x - y)
        rmse = np.sqrt(np.mean((x - y) ** 2))
        mae = np.mean(np.abs(x - y))
        sigma_ratio = np.std(x) / (np.std(y) + 1e-10)  # 避免除零

        # 5. 互相关与滞后
        max_lag = min(180, N // 2)
        cc_max, lag_days = self.cross_correlation(x, y)

        # 返回结果
        results = {
            'station_i': station1_name,
            'station_j': station2_name,
            'N': N,
            'distance_km': distance_km,
            'pearson_r': r_pearson,
            'pearson_p_classic': p_classic,
            'r_ci_lower': r_ci_lo,
            'r_ci_upper': r_ci_hi,
            'p_permutation': p_perm,
            'spearman_rho': rho_spearman,
            'spearman_p': p_spearman,
            'bias': bias,
            'rmse': rmse,
            'mae': mae
        }

        return results

    def plot_pairwise_comparison(self, df1, df2, results,
                                 date_col='YYYYMMDD', value_col='U(m)',
                                 save_path=None):
        """
        绘制成对站点对比图（时序图+散点图+互相关图）
        """
        # 时间对齐
        df1[date_col] = pd.to_datetime(df1[date_col], format='%Y-%m-%d')
        df2[date_col] = pd.to_datetime(df2[date_col], format='%Y-%m-%d')
        merged = pd.merge(df1[[date_col, value_col]],
                          on=date_col, suffixes=('_1', '_2')).dropna()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 时序对比
        ax = axes[0, 0]
        ax.plot(merged[date_col], merged[f'{value_col}_1'],
                label=results['station_i'], alpha=0.7)
        ax.plot(merged[date_col], merged[f'{value_col}_2'],
                label=results['station_j'], alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('U(mm)')
        ax.set_title('Time Series Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 散点图+回归线
        ax = axes[0, 1]
        x = merged[f'{value_col}_1'].values
        y = merged[f'{value_col}_2'].values
        ax.scatter(x, y, alpha=0.5, s=10)

        # 拟合线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', lw=2,
                label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        ax.set_xlabel(f'{results["station_i"]} (mm)')
        ax.set_ylabel(f'{results["station_j"]} (mm)')
        ax.set_title(f'Scatter Plot (r={results["pearson_r"]:.3f}, p={results["p_permutation"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 互相关图
        ax = axes[1, 0]
        max_lag = min(180, len(x) // 2)
        x_norm = (x - np.mean(x)) / (np.std(x) * len(x))
        y_norm = (y - np.mean(y)) / np.std(y)
        correlation = signal.correlate(y_norm, x_norm, mode='full')
        lags = signal.correlation_lags(len(x), len(y), mode='full')
        mask = np.abs(lags) <= max_lag

        ax.plot(lags[mask], correlation[mask])
        ax.axvline(results['lag_days'], color='r', linestyle='--',
                   label=f'Max at lag={results["lag_days"]} days')
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'Cross-correlation (max={results["cc_max"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 指标汇总表
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
        Spatial Correlation Metrics
        {'=' * 40}
        Sample Size (N):           {results['N']}
        Distance:                  {results['distance_km']:.2f} km

        Correlation:
          Pearson r:               {results['pearson_r']:.4f}
          95% CI:                  [{results['r_ci_lower']:.4f}, {results['r_ci_upper']:.4f}]
          p (classic):             {results['pearson_p_classic']:.4e}
          p (permutation):         {results['p_permutation']:.4f}
          Spearman ρ:              {results['spearman_rho']:.4f}

        Amplitude Difference:
          Bias:                    {results['bias']:.4f} mm
          RMSE:                    {results['rmse']:.4f} mm
          MAE:                     {results['mae']:.4f} mm
          σ ratio:                 {results['sigma_ratio']:.4f}

        Cross-correlation:
          Max CC:                  {results['cc_max']:.4f}
          Lag at max:              {results['lag_days']} days
        """
        ax.text(0.1, 0.5, metrics_text, fontfamily='monospace',
                fontsize=10, verticalalignment='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图已保存至: {save_path}")

        plt.show()

    def analyze_all_pairs(self, stations_data, stations_info,
                          target_station=None,
                          date_col='YYYYMMDD', value_col='U(m)',
                          detrend=False, min_samples=30):

        results_list = []

        station_names = list(stations_data.keys())

        if target_station:
            # 只分析目标站与其他站
            if target_station not in station_names:
                raise ValueError(f"目标站点 {target_station} 不在数据中")

            pairs = [(target_station, s) for s in station_names if s != target_station]
            print(f"分析目标站点 {target_station} 与其他 {len(pairs)} 个站点...")
        else:
            # 分析所有站点对
            pairs = [(station_names[i], station_names[j])
                     for i in range(len(station_names))
                     for j in range(i + 1, len(station_names))]
            print(f"分析所有站点对，共 {len(pairs)} 对...")

        for idx, (s1, s2) in enumerate(pairs):
            print(f"  [{idx + 1}/{len(pairs)}] {s1} - {s2}...", end=' ')

            try:
                result = self.analyze_pair(
                    stations_data[s1], stations_data[s2],
                    station1_name=s1, station2_name=s2,
                    lat1=stations_info[s1]['lat'],
                    lon1=stations_info[s1]['lon'],
                    lat2=stations_info[s2]['lat'],
                    lon2=stations_info[s2]['lon'],
                    date_col=date_col,
                    value_col=value_col,
                    detrend=detrend
                )

                if result['N'] >= min_samples:
                    results_list.append(result)
                    print(f"✓ (N={result['N']}, r={result['pearson_r']:.3f})")
                else:
                    print(f"⊗ 样本不足 (N={result['N']} < {min_samples})")

            except Exception as e:
                print(f"✗ 错误: {e}")

        results_df = pd.DataFrame(results_list)

        if len(results_df) > 0:
            # 按相关系数降序排列
            results_df = results_df.sort_values('pearson_r', ascending=False)

        return results_df

    def select_best_neighbors(self, results_df, target_station,
                              min_r=0.7, max_distance_km=100,
                              max_p=0.05, min_N=60):
        """
        根据筛选条件选出最优邻站

        参数:
            results_df: analyze_all_pairs返回的DataFrame
            target_station: 目标站点名
            min_r: 最小Pearson相关系数
            max_distance_km: 最大距离（km）
            max_p: 最大p值（permutation）
            min_N: 最小样本数

        返回: 筛选后的DataFrame，按综合评分排序
        """
        # 筛选与目标站相关的记录
        mask = (results_df['station_i'] == target_station) | \
               (results_df['station_j'] == target_station)
        df_target = results_df[mask].copy()

        # 应用筛选条件
        df_filtered = df_target[
            (df_target['pearson_r'] >= min_r) &
            (df_target['distance_km'] <= max_distance_km) &
            (df_target['p_permutation'] <= max_p) &
            (df_target['N'] >= min_N)
            ].copy()

        # 按质量评分降序
        df_filtered = df_filtered.sort_values('quality_score', ascending=False)

        print(f"\n目标站点 {target_station} 的邻站筛选结果:")
        print(f"  筛选条件: r≥{min_r}, d≤{max_distance_km}km, p≤{max_p}, N≥{min_N}")
        print(f"  符合条件的邻站数: {len(df_filtered)}")

        return df_filtered

    def plot_correlation_heatmap(self, results_df, save_path=None):
        """绘制相关系数热图矩阵"""
        # 构建相关矩阵
        stations = sorted(list(set(results_df['station_i'].tolist() +
                                   results_df['station_j'].tolist())))
        n = len(stations)
        corr_matrix = np.ones((n, n))

        for _, row in results_df.iterrows():
            i = stations.index(row['station_i'])
            j = stations.index(row['station_j'])
            corr_matrix[i, j] = row['pearson_r']
            corr_matrix[j, i] = row['pearson_r']

        # 绘图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f',
                    xticklabels=stations, yticklabels=stations,
                    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                    cbar_kws={'label': 'Pearson r'})
        plt.title('Spatial Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热图已保存至: {save_path}")
        plt.show()

    def plot_distance_vs_correlation(self, results_df, target_station=None,
                                     save_path=None):
        """绘制距离-相关性散点图"""
        df_plot = results_df.copy()

        if target_station:
            mask = (df_plot['station_i'] == target_station) | \
                   (df_plot['station_j'] == target_station)
            df_plot = df_plot[mask]

        plt.figure(figsize=(10, 6))

        # 散点图，按p值着色
        scatter = plt.scatter(df_plot['distance_km'], df_plot['pearson_r'],
                              c=df_plot['p_permutation'], cmap='RdYlGn_r',
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # 添加站点标签
        for _, row in df_plot.iterrows():
            neighbor = row['station_j'] if row['station_i'] == target_station else row['station_i']
            plt.annotate(neighbor,
                         xy=(row['distance_km'], row['pearson_r']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.7)

        plt.colorbar(scatter, label='p-value (permutation)')
        plt.xlabel('Distance (km)', fontsize=12)
        plt.ylabel('Pearson r', fontsize=12)

        title = f'Distance vs Correlation'
        if target_station:
            title += f' (Target: {target_station})'
        plt.title(title, fontsize=14, fontweight='bold')

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"散点图已保存至: {save_path}")
        plt.show()


# ==================== 使用示例 ====================

def example_usage_single_pair():
    """示例1: 单对站点分析"""
    analyzer = GNSSCorrelationAnalysis(nboot=2000, nperm=2000)

    df_ynys = pd.read_csv('D:/Grade 1/GNSS-LSTM+Attention+SG/Spatial Correlation - Machine Learning/GSAX/GSAX.csv')
    df_ynmh = pd.read_csv('D:/Grade 1/GNSS-LSTM+Attention+SG/Spatial Correlation - Machine Learning/GSAX/GSDH.csv')

    stations_info = {
        'GSAX': {'lat': 40.52, 'lon': 95.76},
        'GSDH': {'lat': 40.14, 'lon': 94.68}
    }

    results = analyzer.analyze_pair(
        df_ynys, df_ynmh,
        station1_name='GSAX', station2_name='GSDH',
        lat1=stations_info['GSAX']['lat'],
        lon1=stations_info['GSAX']['lon'],
        lat2=stations_info['GSDH']['lat'],
        lon2=stations_info['GSDH']['lon'],
        date_col='YYYYMMDD', value_col='U(m)',
        detrend=False
    )

    print(f"Pearson r: {results['pearson_r']:.4f}")
    print(f"p-value: {results['p_permutation']:.4f}")

    # 添加可视化
    print("\n生成可视化图表...")
    analyzer.plot_pairwise_comparison(
        df_ynys, df_ynmh, results,
        date_col='YYYYMMDD', value_col='U(m)',
        save_path='GSAX_GSDH_comparison.png'
    )

    # 打印详细结果
    print("\n" + "=" * 50)
    print("详细分析结果:")
    print("=" * 50)
    print(f"站点对: {results['station_i']} - {results['station_j']}")
    print(f"共同样本数: {results['N']}")
    print(f"距离: {results['distance_km']:.2f} km")
    print(f"Pearson相关系数: {results['pearson_r']:.4f}")
    print(f"95%置信区间: [{results['r_ci_lower']:.4f}, {results['r_ci_upper']:.4f}]")
    print(f"经典p值: {results['pearson_p_classic']:.4e}")
    print(f"置换检验p值: {results['p_permutation']:.4f}")
    print(f"Spearman相关系数: {results['spearman_rho']:.4f}")
    print(f"偏置(Bias): {results['bias']:.4f} mm")
    print(f"均方根误差(RMSE): {results['rmse']:.4f} mm")
    print(f"平均绝对误差(MAE): {results['mae']:.4f} mm")
    print(f"标准差比率: {results['sigma_ratio']:.4f}")
    print(f"最大互相关系数: {results['cc_max']:.4f}")
    print(f"最大相关滞后天数: {results['lag_days']} days")

    return results



if __name__ == "__main__":
    print("=" * 60)
    print("GNSS空间相关性分析工具")
    print("=" * 60)

    # 询问用户要执行哪种分析
    choice = input("\n请选择分析模式:\n  [1] 单对站点分析\n  [2] 多站点批量分析\n请输入选择 (1 或 2): ").strip()

    if choice == "1":
        print("\n开始单对站点分析...")
        results = example_usage_single_pair()
        print(f"\n分析完成！Pearson r: {results['pearson_r']:.4f}")

    elif choice == "2":
        print("\n开始多站点批量分析...")
        results_df, best_neighbors = example_usage_multi_stations()
        if len(results_df) > 0:
            print(f"\n批量分析完成！共分析 {len(results_df)} 对站点")
            if len(best_neighbors) > 0:
                print(f"找到 {len(best_neighbors)} 个符合条件的邻站")
            else:
                print("未找到符合条件的邻站")
        else:
            print("批量分析完成，但未找到有效的站点对")

    else:
        print("无效选择，请重新运行并输入 1 或 2")

    print("=" * 60)