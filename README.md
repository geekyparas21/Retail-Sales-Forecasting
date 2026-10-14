# Rossmann Store Sales Forecasting

An end-to-end machine learning project to forecast daily sales for Rossmann drug stores at both the **individual store level** and **national aggregate level**.

## Project Overview
This project evaluates and compares different time-series forecasting methodologies—specifically **XGBoost (with recursive multi-step forecasting)** and **Meta Prophet**—to predict future sales while strictly avoiding data leakage.

### Key Highlights
*   **Store-Level Forecasting:** Models daily sales for individual stores using historical sales, calendar features, promotional schedules, and lag/rolling statistics.
*   **National-Level Forecasting:** Aggregates individual store sales to model national demand trends using macro-indicators (e.g., overall store open rates and active promotions).
*   **Leakage-Free Validation:** Implements a custom recursive framework for tree-based models so that predictions are strictly built on prior predictions rather than true future values.

---

## Dataset
The dataset is sourced from the [Kaggle Rossmann Store Sales Competition](https://www.kaggle.com/c/rossmann-store-sales). It contains historical sales data for 1,115 Rossmann stores.

---

## Final Results

### Store-Level Performance (Store 1)
| Model | MAE | RMSE | WMAPE | Forecast Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **260.11** | **360.35** | **6.92%** | **93.08%** |
| Log-XGBoost | 271.32 | 386.14 | 7.22% | 92.78% |
| Prophet | 295.14 | 406.19 | 7.85% | 92.15% |

### National-Level Performance
| Model | MAE | RMSE | WMAPE | Forecast Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| XGBoost | 568,621.82 | 852,119.77 | 8.36% | 91.64% |
| **Prophet** | **460,809.07** | **625,853.03** | **6.78%** | **93.22%** |

* **XGBoost** excels at learning complex, localized, step-by-step relationships at the individual store level.
* **Prophet** captures aggregate national seasonality, trends, and macroeconomic holidays much more robustly.

---

## Getting Started

### Prerequisites
Install all library dependencies using:
```bash
pip install -r requirements.txt
```

### How to Run
1. Open `Rossmann_Sales_Forecasting.ipynb` in Google Colab or your local Jupyter environment.
2. Ensure your Kaggle API credentials are authenticated to download the competition data.
3. Run all cells sequentially to reproduce the preprocessing, feature engineering, model training, and evaluations.