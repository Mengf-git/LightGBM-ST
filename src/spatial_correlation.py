"""
GNSS Station Spatial Correlation Analysis - SCI Paper-Grade Code
Core Features:
1. Temporal alignment and common sample count statistics
2. Haversine distance calculation
3. Pearson/Spearman correlation coefficients (with bootstrap CI and permutation test)
4. Amplitude difference metrics (Bias, RMSE, MAE, sigma ratio)
5. Cross-correlation and lag analysis
6. Output complete pairwise_metrics.csv and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(0)


class GNSSCorrelationAnalysis:
    """GNSS Station Spatial Correlation Analysis Class"""

    def __init__(self, nboot=2000, nperm=2000):
        """
        Parameters:
            nboot: Number of bootstrap resampling iterations
            nperm: Number of permutation test iterations
        """
        self.nboot = nboot
        self.nperm = nperm

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate Haversine spherical distance (unit: km)

        Formula:
            Δφ = φ2 - φ1
            Δλ = λ2 - λ1
            a = sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2)
            d = 2R·arcsin(√a), R=6371 km
        """
        R = 6371.0  # Earth radius (km)

        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def bootstrap_correlation(self, x, y):
        """
        Compute 95% confidence interval for Pearson correlation using bootstrap

        Returns: (mean r, CI lower bound, CI upper bound)
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
        Compute p-value using permutation test

        H0: No correlation between x and y
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
        Compute cross-correlation and find maximum correlation with corresponding lag

        Parameters:
            max_lag: Maximum lag in days (recommended <= N/2)

        Returns: (maximum cross-correlation value, corresponding lag in days)
        """
        # Normalize series
        x_norm = (x - np.mean(x)) / (np.std(x) * len(x))
        y_norm = (y - np.mean(y)) / np.std(y)

        # Compute cross-correlation
        correlation = signal.correlate(y_norm, x_norm, mode='full')
        lags = signal.correlation_lags(len(x), len(y), mode='full')

        # Restrict to max_lag range
        mask = np.abs(lags) <= max_lag
        correlation = correlation[mask]
        lags = lags[mask]

        # Find maximum value
        max_idx = np.argmax(np.abs(correlation))
        cc_max = correlation[max_idx]
        lag_at_max = lags[max_idx]

        return cc_max, lag_at_max

    def detrend_series(self, series, method='linear'):
        """
        Detrend a time series

        Parameters:
            method: 'linear' or 'constant'
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
        Analyze spatial correlation for a single station pair

        Parameters:
            df1, df2: DataFrames for two stations (must contain date_col and value_col)
            station1_name, station2_name: Station names
            lat1, lon1, lat2, lon2: Latitude and longitude of both stations
            date_col: Name of date column
            value_col: Name of value column (e.g., vertical displacement)
            detrend: Whether to apply detrending

        Returns: dict containing all metrics
        """
        # 1. Temporal alignment (intersection)
        df1[date_col] = pd.to_datetime(df1[date_col], format='%Y-%m-%d')
        df2[date_col] = pd.to_datetime(df2[date_col], format='%Y-%m-%d')

        merged = pd.merge(df1[[date_col, value_col]],
                          df2[[date_col, value_col]],
                          on=date_col,
                          suffixes=('_1', '_2'))

        # Remove NaN
        merged = merged.dropna()
        N = len(merged)

        if N < 30:
            print(f"Warning: {station1_name}-{station2_name} has only {N} common samples; results may be unreliable")

        x = merged[f'{value_col}_1'].values
        y = merged[f'{value_col}_2'].values

        # Detrend (optional)
        if detrend:
            x = self.detrend_series(x)
            y = self.detrend_series(y)

        # 2. Haversine distance
        distance_km = self.haversine_distance(lat1, lon1, lat2, lon2)

        # 3. Pearson correlation (with classic p-value, bootstrap CI, permutation p)
        r_pearson, p_classic = stats.pearsonr(x, y)
        r_mean, r_ci_lo, r_ci_hi = self.bootstrap_correlation(x, y)
        p_perm = self.permutation_test(x, y)

        # 4. Spearman correlation (robust to outliers)
        rho_spearman, p_spearman = stats.spearmanr(x, y)

        # 5. Amplitude difference metrics
        bias = np.mean(x - y)
        rmse = np.sqrt(np.mean((x - y) ** 2))
        mae = np.mean(np.abs(x - y))
        sigma_ratio = np.std(x) / (np.std(y) + 1e-10)  # Avoid division by zero

        # 6. Cross-correlation and lag
        max_lag = min(180, N // 2)
        cc_max, lag_days = self.cross_correlation(x, y, max_lag=max_lag)

        # Return results
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
        Plot pairwise station comparison (time series + scatter plot + cross-correlation plot)
        """
        # Temporal alignment
        df1[date_col] = pd.to_datetime(df1[date_col], format='%Y-%m-%d')
        df2[date_col] = pd.to_datetime(df2[date_col], format='%Y-%m-%d')
        merged = pd.merge(df1[[date_col, value_col]],
                          df2[[date_col, value_col]],
                          on=date_col, suffixes=('_1', '_2')).dropna()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Time series comparison
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

        # 2. Scatter plot + regression line
        ax = axes[0, 1]
        x = merged[f'{value_col}_1'].values
        y = merged[f'{value_col}_2'].values
        ax.scatter(x, y, alpha=0.5, s=10)

        # Fit line
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

        # 3. Cross-correlation plot
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

        # 4. Metrics summary table
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
            print(f"Figure saved to: {save_path}")

        plt.show()

    def analyze_all_pairs(self, stations_data, stations_info,
                          target_station=None,
                          date_col='YYYYMMDD', value_col='U(m)',
                          detrend=False, min_samples=30):
        """
        Batch analysis of spatial correlation for all station pairs

        Parameters:
            stations_data: dict, {station_name: DataFrame}
            stations_info: dict, {station_name: {'lat': ..., 'lon': ...}}
            target_station: str, if specified, only analyze this station against all others
            min_samples: int, minimum common sample count threshold

        Returns: DataFrame containing metrics for all station pairs
        """
        results_list = []

        station_names = list(stations_data.keys())

        if target_station:
            # Only analyze target station against others
            if target_station not in station_names:
                raise ValueError(f"Target station {target_station} not found in data")

            pairs = [(target_station, s) for s in station_names if s != target_station]
            print(f"Analyzing target station {target_station} against {len(pairs)} other stations...")
        else:
            # Analyze all station pairs
            pairs = [(station_names[i], station_names[j])
                     for i in range(len(station_names))
                     for j in range(i + 1, len(station_names))]
            print(f"Analyzing all station pairs, total: {len(pairs)} pairs...")

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
                    print(f"⊗ Insufficient samples (N={result['N']} < {min_samples})")

            except Exception as e:
                print(f"✗ Error: {e}")

        results_df = pd.DataFrame(results_list)

        if len(results_df) > 0:
            # Sort by correlation coefficient in descending order
            results_df = results_df.sort_values('pearson_r', ascending=False)

        return results_df

    def select_best_neighbors(self, results_df, target_station,
                              min_r=0.7, max_distance_km=100,
                              max_p=0.05, min_N=60):
        """
        Select optimal neighboring stations based on filtering criteria

        Parameters:
            results_df: DataFrame returned by analyze_all_pairs
            target_station: Target station name
            min_r: Minimum Pearson correlation coefficient
            max_distance_km: Maximum distance (km)
            max_p: Maximum p-value (permutation)
            min_N: Minimum sample count

        Returns: Filtered DataFrame sorted by composite quality score
        """
        # Filter records related to the target station
        mask = (results_df['station_i'] == target_station) | \
               (results_df['station_j'] == target_station)
        df_target = results_df[mask].copy()

        # Apply filtering criteria
        df_filtered = df_target[
            (df_target['pearson_r'] >= min_r) &
            (df_target['distance_km'] <= max_distance_km) &
            (df_target['p_permutation'] <= max_p) &
            (df_target['N'] >= min_N)
            ].copy()

        # Compute composite quality score (weights adjustable)
        # Quality score = r / (distance + epsilon), normalized to 0-1
        epsilon = 1.0  # Avoid division by zero
        df_filtered['quality_score'] = (
                df_filtered['pearson_r'] / (df_filtered['distance_km'] + epsilon)
        )

        # Sort by quality score in descending order
        df_filtered = df_filtered.sort_values('quality_score', ascending=False)

        print(f"\nNeighbor selection results for target station {target_station}:")
        print(f"  Filter criteria: r>={min_r}, d<={max_distance_km}km, p<={max_p}, N>={min_N}")
        print(f"  Number of qualifying neighbors: {len(df_filtered)}")

        return df_filtered

    def plot_correlation_heatmap(self, results_df, save_path=None):
        """Plot correlation coefficient heatmap matrix"""
        # Build correlation matrix
        stations = sorted(list(set(results_df['station_i'].tolist() +
                                   results_df['station_j'].tolist())))
        n = len(stations)
        corr_matrix = np.ones((n, n))

        for _, row in results_df.iterrows():
            i = stations.index(row['station_i'])
            j = stations.index(row['station_j'])
            corr_matrix[i, j] = row['pearson_r']
            corr_matrix[j, i] = row['pearson_r']

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f',
                    xticklabels=stations, yticklabels=stations,
                    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                    cbar_kws={'label': 'Pearson r'})
        plt.title('Spatial Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        plt.show()

    def plot_distance_vs_correlation(self, results_df, target_station=None,
                                     save_path=None):
        """Plot distance vs. correlation scatter plot"""
        df_plot = results_df.copy()

        if target_station:
            mask = (df_plot['station_i'] == target_station) | \
                   (df_plot['station_j'] == target_station)
            df_plot = df_plot[mask]

        plt.figure(figsize=(10, 6))

        # Scatter plot colored by p-value
        scatter = plt.scatter(df_plot['distance_km'], df_plot['pearson_r'],
                              c=df_plot['p_permutation'], cmap='RdYlGn_r',
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add station labels
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
            print(f"Scatter plot saved to: {save_path}")
        plt.show()


# ==================== Usage Examples ====================

def example_usage_single_pair():
    """Example 1: Single station pair analysis"""
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

    # Generate visualization
    print("\nGenerating visualization charts...")
    analyzer.plot_pairwise_comparison(
        df_ynys, df_ynmh, results,
        date_col='YYYYMMDD', value_col='U(m)',
        save_path='GSAX_GSDH_comparison.png'
    )

    # Print detailed results
    print("\n" + "=" * 50)
    print("Detailed Analysis Results:")
    print("=" * 50)
    print(f"Station pair: {results['station_i']} - {results['station_j']}")
    print(f"Common sample count: {results['N']}")
    print(f"Distance: {results['distance_km']:.2f} km")
    print(f"Pearson correlation coefficient: {results['pearson_r']:.4f}")
    print(f"95% confidence interval: [{results['r_ci_lower']:.4f}, {results['r_ci_upper']:.4f}]")
    print(f"Classic p-value: {results['pearson_p_classic']:.4e}")
    print(f"Permutation test p-value: {results['p_permutation']:.4f}")
    print(f"Spearman correlation coefficient: {results['spearman_rho']:.4f}")
    print(f"Bias: {results['bias']:.4f} mm")
    print(f"Root Mean Square Error (RMSE): {results['rmse']:.4f} mm")
    print(f"Mean Absolute Error (MAE): {results['mae']:.4f} mm")
    print(f"Standard deviation ratio: {results['sigma_ratio']:.4f}")
    print(f"Maximum cross-correlation coefficient: {results['cc_max']:.4f}")
    print(f"Lag at maximum cross-correlation: {results['lag_days']} days")

    return results


def example_usage_multi_stations():
    """
    Example 2: Multi-station batch analysis + automatic best neighbor selection

    Data preparation:
    1. Place all station data in the same directory, named e.g.: YNYS.csv, YNMH.csv, ...
    2. Prepare station info table: stations_info.csv (columns: station, lat, lon)
    """

    # === Step 1: Load all station data ===
    import glob

    # Automatically read all CSV files
    data_files = glob.glob('csv/*.csv')  # Update to your csv folder path
    stations_data = {}

    for file in data_files:
        station_name = file.replace('.csv', '').split('/')[-1]
        if station_name != 'stations_info':  # Exclude metadata file
            df = pd.read_csv(file)
            # Check for required columns
            if 'YYYYMMDD' in df.columns and 'U(m)' in df.columns:
                stations_data[station_name] = df
            else:
                print(f"Warning: {file} is missing YYYYMMDD or U(m) column")

    # Station metadata (lat/lon)
    stations_info = {
        'YNYS': {'lat': 26.68, 'lon': 100.75},
        'YNLJ': {'lat': 26.7, 'lon': 100.03},
    }

    # === Step 2: Initialize analyzer ===
    analyzer = GNSSCorrelationAnalysis(nboot=2000, nperm=2000)

    # === Step 3: Batch analysis (with YNYS as target station) ===
    print("\n" + "=" * 60)
    print("Starting batch analysis...")
    print("=" * 60)

    results_df = analyzer.analyze_all_pairs(
        stations_data=stations_data,
        stations_info=stations_info,
        target_station='YNYS',  # Only analyze YNYS against other stations
        date_col='YYYYMMDD',
        value_col='U(m)',
        detrend=False,
        min_samples=30
    )

    # === Step 4: Save complete results ===
    results_df.to_csv('pairwise_metrics_all.csv', index=False)
    print(f"\nComplete results saved to: pairwise_metrics_all.csv")

    # === Step 5: Select best neighboring stations ===
    print("\n" + "=" * 60)
    print("Selecting best neighboring stations...")
    print("=" * 60)

    best_neighbors = analyzer.select_best_neighbors(
        results_df=results_df,
        target_station='YNYS',
        min_r=0.6,          # Correlation coefficient >= 0.6
        max_distance_km=100, # Distance <= 100 km
        max_p=0.05,          # p-value <= 0.05
        min_N=60             # Sample count >= 60
    )

    # Display top 5
    print("\nTop 5 best neighboring stations:")
    print(best_neighbors[['station_j', 'distance_km', 'pearson_r',
                          'p_permutation', 'N', 'quality_score']].head(5))

    # Save filtered results
    best_neighbors.to_csv('best_neighbors_YNYS.csv', index=False)
    print(f"\nBest neighbor list saved to: best_neighbors_YNYS.csv")

    # === Step 6: Visualization ===
    # 6.1 Correlation heatmap
    analyzer.plot_correlation_heatmap(
        results_df,
        save_path='correlation_heatmap.png'
    )

    # 6.2 Distance vs. correlation scatter plot
    analyzer.plot_distance_vs_correlation(
        results_df,
        target_station='YNYS',
        save_path='distance_vs_correlation_YNYS.png'
    )

    # 6.3 Detailed comparison plot (best pair)
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
    print("GNSS Spatial Correlation Analysis Tool")
    print("=" * 60)

    choice = input("\nSelect analysis mode:\n  [1] Single station pair analysis\n  [2] Multi-station batch analysis\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        print("\nStarting single station pair analysis...")
        results = example_usage_single_pair()
        print(f"\nAnalysis complete! Pearson r: {results['pearson_r']:.4f}")

    elif choice == "2":
        print("\nStarting multi-station batch analysis...")
        results_df, best_neighbors = example_usage_multi_stations()
        if len(results_df) > 0:
            print(f"\nBatch analysis complete! Analyzed {len(results_df)} station pairs")
            if len(best_neighbors) > 0:
                print(f"Found {len(best_neighbors)} qualifying neighboring stations")
            else:
                print("No qualifying neighboring stations found")
        else:
            print("Batch analysis complete, but no valid station pairs were found")

    else:
        print("Invalid choice. Please re-run and enter 1 or 2")

    print("=" * 60)
