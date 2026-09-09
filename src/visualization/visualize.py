"""
Publication figure visualization module.
Clean, descriptive plotting functions for forecast curves, regime error distributions,
and residual correlation matrices.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 300


def plot_final_forecast(test_df: pd.DataFrame, y3_preds: np.ndarray, output_dir: str):
    """
    Plots the full annual load forecast time series for Year 3.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    y3_time = test_df['Date'] + pd.to_timedelta(test_df['Hour'] - 1, unit='h')
    ax.plot(y3_time, y3_preds, color='#1f77b4', linewidth=0.9, label='Load Forecast (CAS Ensemble)')
    ax.set_title('Dự báo phụ tải điện cả năm trên tập kiểm thử Năm 3 (Năm 2022)', fontsize=12, pad=10, fontweight='bold')
    ax.set_xlabel('Thời gian (Date)')
    ax.set_ylabel('Phụ tải (MW)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'Figure_6_Final_Forecast.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_monthly_forecast(test_df: pd.DataFrame, y3_preds: np.ndarray, output_dir: str):
    """
    Plots 12 monthly load forecast panels with 24-hour moving averages.
    """
    setup_style()
    df_plot = test_df.copy()
    df_plot['Pred_Load'] = y3_preds
    df_plot['DateTime'] = df_plot['Date'] + pd.to_timedelta(df_plot['Hour'] - 1, unit='h')
    df_plot = df_plot.sort_values('DateTime').reset_index(drop=True)
    df_plot['MA24'] = df_plot['Pred_Load'].rolling(window=24, min_periods=1, center=True).mean()

    month_names = ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
                   'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12']
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharey=True)
    axes = axes.flatten()

    for m in range(1, 13):
        ax = axes[m - 1]
        m_df = df_plot[df_plot['Month'] == m]
        ax.plot(m_df['Day'] + (m_df['Hour']-1)/24.0, m_df['Pred_Load'], color='#1f77b4', alpha=0.6, linewidth=0.7, label='Dự báo từng giờ')
        ax.plot(m_df['Day'] + (m_df['Hour']-1)/24.0, m_df['MA24'], color='#d62728', linewidth=1.2, label='Trung bình trượt 24h')
        ax.set_title(month_names[m - 1], fontsize=10)
        ax.set_xlabel('Ngày trong tháng', fontsize=8)
        ax.set_ylabel('Phụ tải (MW)', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        if m == 1: ax.legend(loc='upper right', fontsize=7)

    fig.suptitle('Đường cong dự báo phụ tải chi tiết 12 tháng (kèm đường trung bình trượt 24 giờ)', fontsize=13, y=0.995, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'Figure_7_Final_Forecast_by_Month.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_conditional_regime_errors(df_regimes: pd.DataFrame, output_dir: str):
    """
    Plots the conditional RMSE comparison across thermal, diurnal, and calendar regimes.
    """
    setup_style()
    short_names = [
        'Nắng nóng cao (CDD>3°C)', 'Lạnh sâu (HDD>5°C)', 'Thời tiết ôn hòa',
        'Giờ cao điểm (15-19h)', 'Giờ thấp điểm (00-06h)', 'Giờ trưa nắng (10-14h)',
        'Ngày trong tuần', 'Cuối tuần / Ngày lễ'
    ]
    df_plot = df_regimes.copy()
    df_plot['Short_Regime'] = short_names[:len(df_plot)]
    x = np.arange(len(df_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - width, df_plot['ElasticNet_RMSE'], width, label='ElasticNet (Tuyến tính)', color='#7f7f7f', alpha=0.85)
    ax.bar(x, df_plot['XGBoost_RMSE'], width, label='XGBoost (Cây quyết định)', color='#1f77b4', alpha=0.9)
    ax.bar(x + width, df_plot['CAS_Ensemble_RMSE'], width, label='Context-Aware Stacking (CAS đề xuất)', color='#2ca02c', alpha=1.0, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Sai số RMSE (MW)', fontsize=11, fontweight='bold')
    ax.set_title('Phân tích sai số RMSE theo 8 điều kiện vận hành và thời tiết thực tế', fontsize=12, pad=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['Short_Regime'], rotation=15, ha='right', fontsize=9.5)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'Figure_8_Conditional_Regime_Errors.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_residual_correlation_matrix(corr_matrix: pd.DataFrame, output_dir: str):
    """
    Plots the pairwise Pearson residual correlation heatmap between base learners.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', vmin=0, vmax=1, fmt='.4f',
                cbar_kws={'label': 'Hệ số tương quan Pearson ($r$)'}, 
                annot_kws={'size': 13, 'weight': 'bold'}, ax=ax)
    ax.set_title('Ma trận tương quan phần dư giữa các mô hình cơ sở\n(Pairwise Residual Correlation Matrix)', 
                 fontsize=10.5, pad=12, fontweight='bold')
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'Figure_Residual_Correlation_Matrix.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
