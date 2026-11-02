"""
Evaluation and Visualization Script for Airfare Price Estimation.
Generates dark-mode compatible, publication-grade figures:
1. Model Comparison (R2, MAE, RMSE)
2. Feature Importance Breakdown
3. Predicted vs Actual Fare Scatter Plot
4. Residuals Distribution
5. Booking Window Price Curve (Days Left vs Fare)
"""

import os
import sys
import json
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_preprocessing import prepare_train_test_data

def style_dark_figure(fig, axes):
    """Applies a consistent sleek dark-mode styling to matplotlib figures."""
    fig.patch.set_facecolor('#0f172a')
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0', labelsize=10)
        ax.xaxis.label.set_color('#f8fafc')
        ax.yaxis.label.set_color('#f8fafc')
        ax.title.set_color('#f8fafc')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(color='#334155', linestyle='--', alpha=0.5)

def generate_visualizations(base_dir: str):
    plots_dir = os.path.join(base_dir, "plots")
    models_dir = os.path.join(base_dir, "models")
    data_path = os.path.join(base_dir, "data", "Clean_Dataset.csv")
    os.makedirs(plots_dir, exist_ok=True)

    plt.style.use('dark_background')

    # 1. Model Comparison Chart
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        df_metrics = pd.DataFrame(metrics_data).T.reset_index().rename(columns={'index': 'Model'})

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        style_dark_figure(fig, axes)

        # R2
        sns.barplot(data=df_metrics, x='Model', y='R2', palette='Blues_r', hue='Model', legend=False, ax=axes[0])
        axes[0].set_title("Model Comparison: R² Score (Higher is Better)", fontweight='bold', pad=12)
        axes[0].set_ylim(0.7, 1.0)
        axes[0].set_ylabel("R² Score")
        axes[0].tick_params(axis='x', rotation=25)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.98),
                             ha='center', va='top', color='#f8fafc', fontweight='bold', fontsize=10)

        # MAE
        sns.barplot(data=df_metrics, x='Model', y='MAE', palette='Oranges_r', hue='Model', legend=False, ax=axes[1])
        axes[1].set_title("Mean Absolute Error (MAE in ₹) (Lower is Better)", fontweight='bold', pad=12)
        axes[1].set_ylabel("MAE (INR)")
        axes[1].tick_params(axis='x', rotation=25)
        for p in axes[1].patches:
            axes[1].annotate(f"₹{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.88),
                             ha='center', va='top', color='#f8fafc', fontweight='bold', fontsize=10)

        # RMSE
        sns.barplot(data=df_metrics, x='Model', y='RMSE', palette='Greens_r', hue='Model', legend=False, ax=axes[2])
        axes[2].set_title("Root Mean Squared Error (RMSE in ₹)", fontweight='bold', pad=12)
        axes[2].set_ylabel("RMSE (INR)")
        axes[2].tick_params(axis='x', rotation=25)
        for p in axes[2].patches:
            axes[2].annotate(f"₹{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.88),
                             ha='center', va='top', color='#f8fafc', fontweight='bold', fontsize=10)

        plt.tight_layout()
        comp_plot_path = os.path.join(plots_dir, "model_comparison.png")
        plt.savefig(comp_plot_path, dpi=300, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {comp_plot_path}")

    # 2. Feature Importance Plot
    feat_path = os.path.join(models_dir, "feature_importance.csv")
    if os.path.exists(feat_path):
        feat_df = pd.read_csv(feat_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        style_dark_figure(fig, ax)

        palette = sns.color_palette("mako", len(feat_df))
        sns.barplot(data=feat_df, x='Importance_Percent', y='Feature', palette=palette, hue='Feature', legend=False, ax=ax)
        ax.set_title("Feature Importance: Drivers of Airfare Pricing (%)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Relative Importance (%)", fontsize=12)
        ax.set_ylabel("Feature Component", fontsize=12)

        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{width:.1f}%",
                        (width + 0.6, p.get_y() + p.get_height() / 2.),
                        va='center', fontsize=11, fontweight='semibold', color='#38bdf8')

        ax.set_xlim(0, max(feat_df['Importance_Percent']) * 1.15)
        plt.tight_layout()
        feat_plot_path = os.path.join(plots_dir, "feature_importance.png")
        plt.savefig(feat_plot_path, dpi=300, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {feat_plot_path}")

    # Load model and test data for residual & predictions plots
    model_path = os.path.join(models_dir, "best_model.joblib")
    if os.path.exists(model_path) and os.path.exists(data_path):
        model = joblib.load(model_path)
        _, X_test, _, y_test, _ = prepare_train_test_data(data_path)

        sample_test = X_test.sample(n=min(10000, len(X_test)), random_state=42)
        sample_y = y_test.loc[sample_test.index]
        preds = model.predict(sample_test)
        residuals = sample_y - preds

        # 3. Predicted vs Actual Plot
        fig, ax = plt.subplots(figsize=(8, 7))
        style_dark_figure(fig, ax)
        ax.scatter(sample_y, preds, alpha=0.35, color='#38bdf8', edgecolors='none', s=22)
        max_val = max(sample_y.max(), preds.max())
        ax.plot([0, max_val], [0, max_val], color='#f43f5e', linestyle='--', linewidth=2, label='Perfect Fit (y = ŷ)')
        ax.set_title("Predicted vs Actual Airfares (INR)", fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel("Actual Fare (₹)", fontsize=12)
        ax.set_ylabel("Predicted Fare (₹)", fontsize=12)
        ax.legend(frameon=True, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
        plt.tight_layout()
        pva_plot_path = os.path.join(plots_dir, "predicted_vs_actual.png")
        plt.savefig(pva_plot_path, dpi=300, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {pva_plot_path}")

        # 4. Residuals Distribution
        fig, ax = plt.subplots(figsize=(9, 5.5))
        style_dark_figure(fig, ax)
        sns.histplot(residuals, bins=60, kde=True, color='#2dd4bf', ax=ax)
        ax.axvline(0, color='#f43f5e', linestyle='--', linewidth=1.8, label='Zero Error')
        ax.set_title("Residual Distribution (Prediction Errors in ₹)", fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel("Residual (Actual - Predicted Fare in ₹)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(frameon=True, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
        plt.tight_layout()
        res_plot_path = os.path.join(plots_dir, "residuals_distribution.png")
        plt.savefig(res_plot_path, dpi=300, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {res_plot_path}")

        # 5. Booking Window Dynamics (Days Left vs Fare)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        style_dark_figure(fig, ax)
        df_plot = sample_test.copy()
        df_plot['Actual_Price'] = sample_y
        df_plot['Predicted_Price'] = preds

        grouped = df_plot.groupby(['days_left', 'class'])[['Actual_Price', 'Predicted_Price']].mean().reset_index()

        sns.lineplot(data=grouped, x='days_left', y='Actual_Price', hue='class', style='class', markers=True, dashes=False, linewidth=2.5, palette=['#38bdf8', '#f59e0b'], ax=ax)
        ax.set_title("Airfare Dynamic Pricing Curve by Booking Window (Days Left)", fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel("Days Left Before Departure (Booking Window)", fontsize=12)
        ax.set_ylabel("Average Ticket Price (₹)", fontsize=12)
        ax.invert_xaxis()
        ax.legend(frameon=True, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
        plt.tight_layout()
        days_plot_path = os.path.join(plots_dir, "days_left_vs_price.png")
        plt.savefig(days_plot_path, dpi=300, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {days_plot_path}")

    print("All visualizations updated with dark-mode styling!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_visualizations(base_dir)
