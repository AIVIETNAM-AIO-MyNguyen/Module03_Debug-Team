"""
SHAP-based 24-Hour Diurnal Feature Importance Analysis.
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def generate_shap_heatmap(forecaster, data_df: pd.DataFrame, feature_cols: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    heatmap_data = pd.DataFrame(index=feature_cols)

    print("Computing SHAP values across all 24 hourly models...")
    for h in range(1, 25):
        hour_data = data_df[data_df['Hour'] == h]
        X_h = hour_data[feature_cols]
        X_h_sample = X_h.sample(n=min(len(X_h), 500), random_state=42)

        model = forecaster.models[h]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_h_sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        heatmap_data[f'Hour_{h}'] = mean_abs_shap

    heatmap_data['Total_Importance'] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values(by='Total_Importance', ascending=False)

    csv_path = os.path.join(output_dir, 'shap_importance_by_hour.csv')
    heatmap_data.to_csv(csv_path)

    heatmap_data = heatmap_data.drop(columns=['Total_Importance'])

    plt.figure(figsize=(16, max(6, len(feature_cols) * 0.35)))
    sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt=".2f", linewidths=.5)
    plt.title('Figure 10: SHAP Feature Importance by Hour of Day (Mean Absolute SHAP Value)', fontsize=13, pad=12)
    plt.xlabel('Hour of Day (1 to 24)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'fig_shap_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"SHAP heatmap saved to: {heatmap_path}")
    return heatmap_data
