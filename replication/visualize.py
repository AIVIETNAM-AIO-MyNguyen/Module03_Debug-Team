"""
Figure generation functions for baseline replication.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 300


def plot_fig1_fig2_load_vs_temp(train_df: pd.DataFrame, output_dir: str):
    setup_style()
    h4_df = train_df[train_df['Hour'] == 4].copy()
    for yr in [1, 2]:
        sub = h4_df[h4_df['Year'] == yr]
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(sub['Site-5 Temp'], sub['Load'], c=sub['Month'], cmap='viridis', s=25, alpha=0.8, edgecolors='k', linewidths=0.2)
        plt.colorbar(sc, ax=ax, label='Month')
        ax.set_title(f"Year {yr} - Hour 4:00 - Site5Temp vs Load")
        ax.set_xlabel('Site5 Temp (°C)')
        ax.set_ylabel('Load')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'Figure_{yr}_Load_vs_Site5Temp_Year{yr}.png')
        plt.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_fig3_pca_variance(train_df: pd.DataFrame, temp_cols: list[str], ghi_cols: list[str], output_dir: str):
    setup_style()
    pca_temp = PCA(n_components=5).fit(train_df[temp_cols])
    pca_ghi = PCA(n_components=5).fit(train_df[ghi_cols])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, 6)
    width = 0.35
    ax.bar(x - width/2, pca_temp.explained_variance_ratio_, width, label='Temperature PCA', color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, pca_ghi.explained_variance_ratio_, width, label='GHI PCA', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('PCA Component Index')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Variance Ratio PCA Comp')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_3_PCA_Components.png'), dpi=300)
    plt.close(fig)


def plot_fig4_weekdays_vs_weekends(train_df: pd.DataFrame, output_dir: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    sample_df = train_df.iloc[:24 * 28].copy()
    time_index = sample_df['Date'] + pd.to_timedelta(sample_df['Hour'] - 1, unit='h')
    ax.plot(time_index, sample_df['Load'], color='#1f77b4', label='Load', linewidth=1.2)
    weekend_mask = sample_df['Is_Weekend'] == 1
    ax.scatter(time_index[weekend_mask], sample_df.loc[weekend_mask, 'Load'], color='#d62728', s=12, label='Weekends', zorder=5)
    ax.set_title('Figure 4: Weekdays vs Weekends from Load data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Load (MW)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_4_Weekdays_vs_Weekends.png'), dpi=300)
    plt.close(fig)


def plot_fig6_final_forecast(test_df: pd.DataFrame, y3_preds: np.ndarray, output_dir: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    y3_time = test_df['Date'] + pd.to_timedelta(test_df['Hour'] - 1, unit='h')
    ax.plot(y3_time, y3_preds, color='blue', linewidth=0.9, label='Load Forecast')
    ax.set_title('Figure 6: Final Forecast (Year 3 / 2022)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Load (MW)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_6_Final_Forecast.png'), dpi=300)
    plt.close(fig)


def plot_fig7_monthly_forecast(test_df: pd.DataFrame, y3_preds: np.ndarray, output_dir: str):
    setup_style()
    df_plot = test_df.copy()
    df_plot['Pred_Load'] = y3_preds
    df_plot['DateTime'] = df_plot['Date'] + pd.to_timedelta(df_plot['Hour'] - 1, unit='h')
    df_plot = df_plot.sort_values('DateTime').reset_index(drop=True)
    df_plot['MA24'] = df_plot['Pred_Load'].rolling(window=24, min_periods=1, center=True).mean()

    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharey=True)
    axes = axes.flatten()

    for m in range(1, 13):
        ax = axes[m - 1]
        m_df = df_plot[df_plot['Month'] == m]
        ax.plot(m_df['Day'] + (m_df['Hour']-1)/24.0, m_df['Pred_Load'], color='blue', alpha=0.6, linewidth=0.7, label='Hourly Forecast')
        ax.plot(m_df['Day'] + (m_df['Hour']-1)/24.0, m_df['MA24'], color='red', linewidth=1.2, label='24h Moving Avg')
        ax.set_title(month_names[m - 1], fontsize=10)
        ax.set_xlabel('Day of Month', fontsize=8)
        ax.set_ylabel('Load (MW)', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        if m == 1: ax.legend(loc='upper right', fontsize=7)

    fig.suptitle('Figure 7: Final Forecast by Month (with 24-hour Moving Average in Red)', fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_7_Final_Forecast_by_Month.png'), dpi=300)
    plt.close(fig)
