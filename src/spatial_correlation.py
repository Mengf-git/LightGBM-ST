"""
GNSS站点间空间相关性分析
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
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "real_GNSS")

warnings.filterwarnings('ignore')

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
        """
        计算Haversine球面距离（单位：km）

        公式:
            Δφ = φ2 - φ1
            Δλ = λ2 - λ1
            a = sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2)
            d = 2R·arcsin(√a), R=6371 km
        """
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
            r_boot[i] = stats.pearsonr(x[idx], y[idx])[0]

        r_mean = np.mean(r_boot)
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
            y_shuffled = np.random.permutation(y)
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
        mask = np.abs(lags) <= max_lag
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
        """
        分析单对站点的空间相关性

        参数:
            df1, df2: 两个站点的DataFrame（必须包含date_col和value_col）
            station1_name, station2_name: 站点名称
            lat1, lon1, lat2, lon2: 两站经纬度
            date_col: 日期列名
            value_col: 数值列名（如vertical位移）
            detrend: 是否去趋势

        返回: dict，包含所有指标
        """
        # 1. 时间对齐（取交集）
        df1[date_col] = pd.to_datetime(df1[date_col], format='%Y-%m-%d')
        df2[date_col] = pd.to_datetime(df2[date_col], format='%Y-%m-%d')

        merged = pd.merge(df1[[date_col, value_col]],
                          df2[[date_col, value_col]],
                          on=date_col,
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
        distance_km = self.haversine_distance(lat1, lon1, lat2, lon2)

        # 3. Pearson相关（含经典p值、bootstrap CI、permutation p）
        r_pearson, p_classic = stats.pearsonr(x, y)
        r_mean, r_ci_lo, r_ci_hi = self.bootstrap_correlation(x, y)
        p_perm = self.permutation_test(x, y)

        # 4. Spearman相关（鲁棒于异常值）
        rho_spearman, p_spearman = stats.spearmanr(x, y)

        # 5. 幅值差异指标
        bias = np.mean(x - y)
        rmse = np.sqrt(np.mean((x - y) ** 2))
        mae = np.mean(np.abs(x - y))
        sigma_ratio = np.std(x) / (np.std(y) + 1e-10)  # 避免除零

        # 6. 互相关与滞后
        max_lag = min(180, N // 2)
        cc_max, lag_days = self.cross_correlation(x, y, max_lag=max_lag)

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
            'mae': mae,
            'sigma_ratio': sigma_ratio,
            'cc_max': cc_max,
            'lag_days': lag_days
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
                          df2[[date_col, value_col]],
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
        """
        批量分析所有站点对的空间相关性

        参数:
            stations_data: dict, {station_name: DataFrame}
            stations_info: dict, {station_name: {'lat': ..., 'lon': ...}}
            target_station: str, 如果指定则只分析该站与其他站的关系
            min_samples: int, 最小共同样本数阈值

        返回: DataFrame，包含所有站点对的指标
        """
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

        # 计算综合评分（可调整权重）
        # 综合评分 = r / (distance + epsilon)，归一化到0-1
        epsilon = 1.0  # 避免除零
        df_filtered['quality_score'] = (
                df_filtered['pearson_r'] / (df_filtered['distance_km'] + epsilon)
        )

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
    
    analyzer = GNSSCorrelationAnalysis(nboot=2000, nperm=2000)

    df_ynys = pd.read_csv(os.path.join(DATA_DIR, "GSAX.csv"))
    df_ynmh = pd.read_csv(os.path.join(DATA_DIR, "GSDH.csv"))


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


    return results


def example_usage_multi_stations():
    """
    多站点批量分析 + 自动筛选最优邻站

    数据准备:
    1. 将所有站点数据放在同一目录
    2. 准备站点信息表
    """

    # === 步骤1: 加载所有站点数据 ===
    import glob

    # 自动读取所有CSV文件
    data_files = glob.glob('csv/*.csv')  # 修改为csv文件夹路径
    stations_data = {}

    for file in data_files:
        station_name = file.replace('.csv', '').split('/')[-1]
        if station_name != 'stations_info':  # 排除元信息文件
            df = pd.read_csv(file)
            # 检查是否包含所需的列
            if 'YYYYMMDD' in df.columns and 'U(m)' in df.columns:
                stations_data[station_name] = df
            else:
                print(f"警告: {file} 缺少YYYYMMDD或U(m)列")

    # 站点元信息（经纬度）
    stations_info = {
        'YNYS': {'lat': 26.68, 'lon': 100.75},
        'YNLJ': {'lat': 26.7, 'lon': 100.03},
    }

    # === 步骤2: 初始化分析器 ===
    analyzer = GNSSCorrelationAnalysis(nboot=2000, nperm=2000)

    # === 步骤3: 批量分析（以YNYS为目标站） ===
    print("\n" + "=" * 60)
    print("开始批量分析...")
    print("=" * 60)

    results_df = analyzer.analyze_all_pairs(
        stations_data=stations_data,
        stations_info=stations_info,
        target_station='YNYS',  # 只分析YNYS与其他站的关系
        date_col='YYYYMMDD',
        value_col='U(m)',
        detrend=False,
        min_samples=30
    )

    # === 步骤4: 保存完整结果 ===
    results_df.to_csv('pairwise_metrics_all.csv', index=False)
    print(f"\n完整结果已保存至: pairwise_metrics_all.csv")

    # === 步骤5: 筛选最优邻站 ===
    print("\n" + "=" * 60)
    print("筛选最优邻站...")
    print("=" * 60)

    best_neighbors = analyzer.select_best_neighbors(
        results_df=results_df,
        target_station='YNYS',
        min_r=0.6,  # 相关系数≥0.6
        max_distance_km=100,  # 距离≤100km
        max_p=0.05,  # p值≤0.05
        min_N=60  # 样本数≥60
    )

    # 显示前5名
    print("\n前5名最优邻站:")
    print(best_neighbors[['station_j', 'distance_km', 'pearson_r',
                          'p_permutation', 'N', 'quality_score']].head(5))

    # 保存筛选结果
    best_neighbors.to_csv('best_neighbors_YNYS.csv', index=False)
    print(f"\n最优邻站列表已保存至: best_neighbors_YNYS.csv")

    # === 步骤6: 可视化 ===
    # 6.1 相关系数热图
    analyzer.plot_correlation_heatmap(
        results_df,
        save_path='correlation_heatmap.png'
    )

    # 6.2 距离-相关性散点图
    analyzer.plot_distance_vs_correlation(
        results_df,
        target_station='YNYS',
        save_path='distance_vs_correlation_YNYS.png'
    )

    # 6.3 详细对比图（选最优的一对）
    if len(best_neighbors) > 0:
        best_neighbor = best_neighbors.iloc[0]
        neighbor_name = best_neighbor['station_j']

        analyzer.plot_pairwise_comparison(
            stations_data['YNYS'],
            stations_data[neighbor_name],
            best_neighbor.to_dict(),
            date_col='YYYYMMDD',
            value_col='U(m)',
            save_path=f'YNYS_{neighbor_name}_detailed.png'
        )

    return results_df, best_neighbors

if __name__ == "__main__":
    print("=" * 60)
    print("GNSS空间相关性分析工具")
    print("=" * 60)
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
