# Project Report: Airfare Component Price Estimation

**Course:** Supervised Learning (INT395)  
**Assignment:** CA1 Project-Based Assignment  
**Institution:** Lovely Professional University (LPU)  
**Topic S.No:** 17  
**Platform / Reference:** Smart India Hackathon (SIH) 2026 — **SIH26056**  

---

## Executive Summary

Domestic airline ticket prices in India are highly volatile, dictated by complex revenue management and dynamic pricing algorithms. Consumers frequently experience sudden price surges, leading to uncertainty when planning travel. 

This project develops an end-to-end Supervised Learning system to solve **SIH 2026 Problem Statement SIH26056: "Airfare Component Price Estimation"**. By leveraging flight booking features—including origin-destination routes, airline carriers, departure/arrival time windows, cabin class, flight duration, and booking windows (`days_left`)—we construct a regression pipeline capable of accurately predicting airfares, decomposing price drivers, and offering an interactive consumer price estimation and booking-sensitivity tool.

Across a benchmark of 5 supervised learning algorithms on **300,153 flight records**, our ensemble models achieved exceptional predictive fidelity, with the **Random Forest Regressor** reaching an **$R^2$ of 0.9743** and a **Mean Absolute Error (MAE) of ₹1,819.17**, closely followed by **HistGradientBoosting** ($R^2 = 0.9724$).

---

## 1. Problem Formulation & Objectives

### 1.1 SIH 2026 Problem Context
- **Topic Name:** Airfare Component Price Estimation
- **Problem Statement:** Estimate airfare from route, airline, travel date, duration, and booking-window features for a consumer price-estimation tool.
- **Expected Deliverables:**
  1. Regression pipeline
  2. Categorical encoding
  3. Model comparison (reporting MAE, RMSE, and $R^2$)
  4. Feature-importance analysis
  5. Interactive fare estimator web interface

### 1.2 Mathematical Formulation
Given an input feature vector $\mathbf{x} = [x_{\text{airline}}, x_{\text{source}}, x_{\text{destination}}, x_{\text{departure\_time}}, x_{\text{arrival\_time}}, x_{\text{stops}}, x_{\text{class}}, x_{\text{route}}, x_{\text{duration}}, x_{\text{days\_left}}]^T$, the objective is to learn a mapping function $f: \mathcal{X} \rightarrow \mathbb{R}^+$ that minimizes the expected loss:

$$\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(y_i, f(\mathbf{x}_i; \theta))$$

where $y_i$ represents the actual ticket price in Indian Rupees (INR) and $\mathcal{L}$ corresponds to Absolute Error ($|y - \hat{y}|$) and Squared Error ($(y - \hat{y})^2$).

---

## 2. Dataset Overview & Feature Engineering

### 2.1 Dataset Profile
The dataset contains **300,153 validated domestic flight listings** across 6 major Indian metropolitan hubs (Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai) spanning two distinct cabin classes (Economy and Business).

| Feature Name | Data Type | Nature | Description / Domain |
| :--- | :--- | :--- | :--- |
| `airline` | Categorical | Nominal | 6 carriers: Vistara, Air India, IndiGo, AirAsia, GO FIRST, SpiceJet |
| `source_city` | Categorical | Nominal | Origin metro: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai |
| `destination_city` | Categorical | Nominal | Destination metro (mutually exclusive with source city) |
| `route` | Categorical | Interaction | Synthetic route interaction feature (e.g. `Delhi -> Mumbai`) |
| `departure_time` | Categorical | Ordinal/Categorical | Time window: Early Morning, Morning, Afternoon, Evening, Night, Late Night |
| `arrival_time` | Categorical | Ordinal/Categorical | Arrival time slot |
| `stops` | Categorical | Ordinal | Number of intermediate stops: `zero`, `one`, `two_or_more` |
| `class` | Categorical | Binary | Cabin category: `Economy` vs `Business` |
| `duration` | Continuous | Numerical | Total flight travel duration in hours (range: 0.83h to 49.83h) |
| `days_left` | Discrete | Numerical | Booking window: Days remaining until flight departure (1 to 49 days) |
| **`price`** | Continuous | **Target** | Airfare in Indian Rupees (₹) |

### 2.2 Data Preprocessing & Categorical Encoding Strategy
1. **Handling Redundant Features:** Dropped arbitrary dataset indices (`Unnamed: 0`) and high-cardinality flight numbers (`flight`) to avoid overfitting to specific tail numbers.
2. **Route Interaction Engineering:** Synthesized the composite categorical feature `route = source_city + ' -> ' + destination_city` to allow the model to capture route-specific supply-demand economics.
3. **Categorical Encoding:** Applied `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` across nominal attributes (`airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `stops`, `class`, `route`). Dropping the first category prevents strict multicollinearity in linear models.
4. **Numerical Standardization:** Applied `StandardScaler` to `duration` and `days_left` to standardize features with zero mean and unit variance.
5. **Data Partitioning:** Preserved an 80/20 train-test split (240,122 training rows, 60,031 test rows) using `random_state=42` for exact reproducibility.

---

## 3. Supervised Learning Model Comparison

We evaluated five candidate regression algorithms across four performance metrics:
- **MAE (Mean Absolute Error):** Average magnitude of errors in ₹.
- **RMSE (Root Mean Squared Error):** Penalizes larger outlier prediction errors.
- **MAPE (Mean Absolute Percentage Error):** Relative percentage error across price bands.
- **$R^2$ (Coefficient of Determination):** Proportion of fare variance explained by features.

### 3.1 Empirical Benchmark Results

| Model Architecture | MAE (₹) | RMSE (₹) | MAPE (%) | $R^2$ Score | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | ₹4,549.50 | ₹6,727.83 | 46.81% | 0.9122 | 1.36s |
| **Ridge Regression ($\alpha=1.0$)** | ₹4,549.49 | ₹6,727.83 | 46.81% | 0.9122 | 0.79s |
| **Decision Tree Regressor** | ₹2,023.24 | ₹4,035.40 | 13.15% | 0.9684 | 2.26s |
| **HistGradientBoosting Regressor** | ₹2,208.53 | ₹3,773.38 | 17.01% | 0.9724 | 9.95s |
| **Random Forest Regressor (Best)** | **₹1,819.17** | **₹3,637.18** | **11.93%** | **0.9743** | 35.27s |

### 3.2 Key Model Observations
1. **Linear Baselines ($R^2 = 0.9122$):** While explaining over 91% of variance due to the strong linear shift caused by cabin class, linear models exhibit high MAPE (46.81%) because they cannot model the sharp non-linear price surge that occurs in the final days before departure.
2. **Decision Trees & Ensembles:** Tree-based non-linear regressors dramatically reduce the MAE from ₹4,549 down to ₹1,819.
3. **Random Forest & HistGradientBoosting:** Both ensemble approaches demonstrate near-parity, with Random Forest achieving the best overall generalization ($R^2 = 0.9743$, MAPE $= 11.93\%$) and HistGradientBoosting providing fast convergence with competitive RMSE (₹3,773.38).

---

## 4. Feature Importance & Component Price Analysis

Using permutation feature importance on held-out test data, we isolated the percentage contribution of each feature to the overall predicted ticket price:

| Rank | Feature | Importance Share (%) | Behavioral & Economic Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `class` | **86.14%** | Cabin class is the single greatest driver. Business class fares trade at 4x–6x higher base multiples than economy class. |
| **2** | `duration` | **6.36%** | Fuel burn, aircraft block-hours, and crew costs scale directly with flight duration. |
| **3** | `airline` | **2.05%** | Captures carrier segmentation: Full-service carriers (Vistara, Air India) command premium fare bands versus budget carriers (IndiGo, SpiceJet, AirAsia). |
| **4** | `days_left` | **1.56%** | Dynamic yield management factor. Fares surge exponentially within 1–7 days before departure. |
| **5** | `route` | **1.28%** | Captures high-demand metro corridors (e.g., Delhi–Mumbai business route) versus secondary corridors. |
| **6** | `destination_city` | **0.90%** | Destination airport slot fees and local demand elasticity. |
| **7** | `source_city` | **0.76%** | Origin airport departure taxes and regional capacity. |
| **8** | `arrival_time` | **0.43%** | Prime landing windows (evening/morning) command minor premiums. |
| **9** | `departure_time` | **0.32%** | Unpopular flight times (late-night "red-eyes") have noticeable discounts. |
| **10** | `stops` | **0.21%** | Non-stop direct flights carry convenience premiums. |

---

## 5. Interactive Fare Estimator Application (Streamlit)

As required by the SIH 2026 project specification, an interactive web application was constructed in `app.py`:

### Key Features of the Web Tool:
1. **Interactive Inputs:** Sliders and drop-downs for all travel parameters (Origin, Destination, Carrier, Cabin Class, Duration, Departure/Arrival Times, and Days Left).
2. **Real-time Price Estimation:** Outputs predicted fare in ₹ INR alongside 68% confidence bounds ($\pm \text{MAE}$).
3. **Dynamic Sensitivity Simulation:** Plots an interactive line chart illustrating how the fare for the selected route evolves as the departure date approaches (from 50 days out to 1 day out).
4. **Model Comparison & Diagnostics Tab:** Displays the benchmark metrics table alongside residual distributions and actual-vs-predicted regression diagnostic plots.
5. **Feature Importance Tab:** Visually communicates key price drivers to help travelers optimize their booking schedules.

---

## 6. How to Run the Project

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `streamlit`

### Execution Steps
```bash
# 1. Navigate to the project root directory
cd "C:\Users\TEJASH KUMAR SINGH\.gemini\antigravity\scratch\airfare-price-estimation"

# 2. Train and benchmark the regression models
python src/train_models.py

# 3. Generate high-resolution evaluation and residual plots
python src/evaluate.py

# 4. Launch the Interactive Streamlit Web Application
streamlit run app.py
```

---

## 7. Conclusion

This project successfully fulfills all deliverables outlined in **INT395 CA1 Project S.No 17 (SIH26056)**:
- Engineered a robust, scalable Supervised Learning regression pipeline.
- Benchmarked linear, tree, and ensemble architectures, establishing Random Forest as the optimal predictor with **97.43% variance explained** and an **MAE of ₹1,819.17**.
- Quantified the impact of cabin class, flight duration, airline carrier tier, and booking window on fare composition.
- Delivered a deployable, consumer-friendly interactive web interface for real-time fare estimation and price sensitivity analysis.
