"""
Visualization module to reproduce all figures from the paper:
- Figure 1 & 2: Load vs Site 5 Temperature (Heating and Cooling tails)
- Figure 3: PCA Variance Ratio
- Figure 4: Weekdays vs Weekends load profile
- Figure 5: XGBoost Test Cases (Year 1->2, Year 2->1, Cross-Validation)
- Figure 6: Final Year 3 Forecast
- Figure 7: Final Forecast by Month (with 24h moving average)
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
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['figure.dpi'] = 300


def plot_fig1_fig2_load_vs_temp(train_df: pd.DataFrame, output_dir: str):
    """
    Reproduces Figure 1 and Figure 2:
    Load vs Site 5 Temperature for Year 1 and Year 2 (Hour 4).
    """
    setup_style()
    h4_df = train_df[train_df['Hour'] == 4].copy()

    for yr in [1, 2]:
        sub = h4_df[h4_df['Year'] == yr]
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(sub['Site-5 Temp'], sub['Load'], c=sub['Month'], cmap='viridis', s=25, alpha=0.8, edgecolors='k', linewidths=0.2)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Month')
        ax.set_title(f"Year {yr} - Hour 4:00 - Site5Temp vs Load")
        ax.set_xlabel('Site5 Temp (°C)')
        ax.set_ylabel('Load')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f'Figure_{yr}_Load_vs_Site5Temp_Year{yr}.png')
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_fig3_pca_variance(train_df: pd.DataFrame, temp_cols: list[str], ghi_cols: list[str], output_dir: str):
    """
    Reproduces Figure 3: PCA Variance Ratio Components.
    """
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

    out_path = os.path.join(output_dir, 'Figure_3_PCA_Components.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig4_weekdays_vs_weekends(train_df: pd.DataFrame, output_dir: str):
    """
    Reproduces Figure 4: Weekdays vs Weekends load profile dip.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    # Subset a representative multi-week window for clarity
    sample_df = train_df.iloc[:24 * 28].copy()  # 4 weeks
    time_index = sample_df['Date'] + pd.to_timedelta(sample_df['Hour'] - 1, unit='h')

    ax.plot(time_index, sample_df['Load'], color='#1f77b4', label='Load', linewidth=1.2)

    # Highlight weekends
    weekend_mask = sample_df['Is_Weekend'] == 1
    ax.scatter(time_index[weekend_mask], sample_df.loc[weekend_mask, 'Load'], color='#d62728', s=12, label='Weekends', zorder=5)

    ax.set_title('Figure 4: Weekdays vs Weekends from Load data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Load (MW)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'Figure_4_Weekdays_vs_Weekends.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig5_xgboost_test_cases(
    y1_df: pd.DataFrame,
    y2_df: pd.DataFrame,
    y2_preds: np.ndarray,
    y1_preds: np.ndarray,
    cv_trues: np.ndarray,
    cv_preds: np.ndarray,
    output_dir: str
):
    """
    Reproduces Figure 5: Test Cases for XGBoost Model (Three subplots).
    """
    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharey=True)

    # Helper time indices
    y1_time = y1_df['Date'] + pd.to_timedelta(y1_df['Hour'] - 1, unit='h')
    y2_time = y2_df['Date'] + pd.to_timedelta(y2_df['Hour'] - 1, unit='h')

    # Subplot 1: Year 1 -> Year 2
    ax1 = axes[0]
    ax1.plot(y1_time, y1_df['Load'], color='blue', alpha=0.7, label='Training Data (Actual)', linewidth=0.8)
    ax1.plot(y2_time, y2_df['Load'], color='green', alpha=0.7, label='Test Data (Actual)', linewidth=0.8)
    ax1.plot(y2_time, y2_preds, color='red', alpha=0.7, label='Test Data (Predicted)', linewidth=0.8)
    r2_1 = r2_score(y2_df['Load'], y2_preds)
    rmse_1 = np.sqrt(np.mean((y2_df['Load'] - y2_preds)**2))
    mape_1 = np.mean(np.abs((y2_df['Load'] - y2_preds) / y2_df['Load'])) * 100
    ax1.text(0.02, 0.85, f"R² Overall: {r2_1:.4f}\nRMSE Overall: {rmse_1:.2f}\nMAPE: {mape_1:.2f}%",
             transform=ax1.transAxes, bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))
    ax1.set_title("Sequential Predictions: Year1 -> Year2")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Year 2 -> Year 1
    ax2 = axes[1]
    ax2.plot(y2_time, y2_df['Load'], color='blue', alpha=0.7, label='Training Data (Actual)', linewidth=0.8)
    ax2.plot(y1_time, y1_df['Load'], color='green', alpha=0.7, label='Test Data (Actual)', linewidth=0.8)
    ax2.plot(y1_time, y1_preds, color='red', alpha=0.7, label='Test Data (Predicted)', linewidth=0.8)
    r2_2 = r2_score(y1_df['Load'], y1_preds)
    rmse_2 = np.sqrt(np.mean((y1_df['Load'] - y1_preds)**2))
    mape_2 = np.mean(np.abs((y1_df['Load'] - y1_preds) / y1_df['Load'])) * 100
    ax2.text(0.02, 0.85, f"R² Overall: {r2_2:.4f}\nRMSE Overall: {rmse_2:.2f}\nMAPE: {mape_2:.2f}%",
             transform=ax2.transAxes, bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))
    ax2.set_title("Sequential Predictions: Year2 -> Year1")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Cross-Validation Both Years
    ax3 = axes[2]
    # For CV plot, align chronological sequence
    n_cv = len(cv_trues)
    cv_time = pd.date_range(start='2020-05-01', periods=n_cv, freq='h')[:n_cv]
    ax3.plot(cv_time, cv_trues, color='green', alpha=0.7, label='Actual', linewidth=0.8)
    ax3.plot(cv_time, cv_preds, color='red', alpha=0.7, label='CV Predictions', linewidth=0.8)
    r2_cv = r2_score(cv_trues, cv_preds)
    rmse_cv = np.sqrt(np.mean((cv_trues - cv_preds)**2))
    mape_cv = np.mean(np.abs((cv_trues - cv_preds) / cv_trues)) * 100
    ax3.text(0.02, 0.85, f"R² CV: {r2_cv:.4f}\nRMSE CV: {rmse_cv:.2f}\nMAPE: {mape_cv:.2f}%",
             transform=ax3.transAxes, bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))
    ax3.set_title("Time Series Cross-Validation Results - Both Years")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'Figure_5_Test_Cases_XGBoost.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig6_final_forecast(test_df: pd.DataFrame, y3_preds: np.ndarray, output_dir: str):
    """
    Reproduces Figure 6: Final Load Forecast (Year 3).
    """
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

    out_path = os.path.join(output_dir, 'Figure_6_Final_Forecast.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig7_monthly_forecast(test_df: pd.DataFrame, y3_preds: np.ndarray, output_dir: str):
    """
    Reproduces Figure 7: Final Forecast by Month (12 subplots with 24-hour moving average).
    """
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
        if m == 1:
            ax.legend(loc='upper right', fontsize=7)

    fig.suptitle('Figure 7: Final Forecast by Month (with 24-hour Moving Average in Red)', fontsize=13, y=0.995)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'Figure_7_Final_Forecast_by_Month.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
