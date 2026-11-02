# ✈️ Enterprise Airfare Component Price Estimation & Yield Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest%20(97.43%25%20R%C2%B2)-success.svg)](https://scikit-learn.org)
[![Google Flights Live](https://img.shields.io/badge/Live%20Data-Google%20Flights%20API-4285F4.svg)](https://www.google.com/travel/flights)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Smart India Hackathon (SIH 2026 — Problem Statement: SIH26056)**  
**Course:** Supervised Learning (INT395) | Lovely Professional University  
**Topic S.No:** 17 — *Airfare Component Price Estimation and Yield Management System*  
**Developer:** [KTejash](https://github.com/KTejash)

---

## 🌟 Executive Overview

Standard academic flight price models typically predict simple flat fares between 5–6 metro cities using toy regression. This project engineers a **full-stack, production-grade enterprise aviation intelligence platform** that combines:

1. **Supervised Machine Learning Core:** Trained on **300,153 validated Indian domestic flights** across 5 competitive regression architectures, achieving **97.43% $R^2$ accuracy** and **₹1,819 MAE**.
2. **Real-Time Live Market Pricing (Google Flights API):** Live OTA price fetcher querying current online bookable fares to benchmark machine learning estimates directly against real-time market dynamics.
3. **Geospatial Navigation Engine:** Covers **50+ commercial airports** across all Indian States & Union Territories with Haversine great-circle geodesics, aircraft block speed modeling, true compass bearing, and carbon footprint ($CO_2$) telemetry.
4. **Interactive Leaflet & Google Maps Flight Simulation:** Satellite Hybrid, Streets, and Terrain tiles featuring curved 3D parabolic flight arcs, animated aircraft simulation, and automatic multi-leg break-journey / layover visualization.
5. **Component Cost Decomposition (Bill of Airfare):** Itemized breakdown into Base Operating Tariff, Fuel & Dynamic Demand Surcharge (YQ), Airport Development Fees (UDF/PSF/ASF), and Statutory GST.
6. **Yield Management Booking Advisory:** Heuristic yield curve simulation analyzing booking window acceleration (50 days down to 1 day prior) to deliver actionable "Buy Now vs. Wait" advice.
7. **Universal 1-Page PDF Quotation Generator:** Device-independent, high-fidelity executive quotation report generated via ReportLab with instant download.

---

## 🚀 Key Features & Capabilities

### 1. 🔴 Real-Time Live Google Flights Integration
- Direct live query engine using `fast-flights` (Protobuf-encoded Google Flights queries without paid API key requirements).
- Fetches live bookable fares, departure/arrival schedules, operating aircraft models (e.g. *Airbus A320neo, Boeing 737MAX*), and stops count.
- **Market Variance Index:** Automatically compares ML predicted baseline vs. live online market fares (e.g. `+3.4% Live Market Surge` or `🎯 Near-Exact ML Match`).
- Built-in **15-minute smart in-memory TTL caching** to prevent rate-limiting while providing instantaneous responses.
- One-click `"🌐 Open Live Search on Google Flights"` deep-link prefilled with origin, destination, and travel date.

### 2. 🛫 Smart Airport Selection (50+ Indian Commercial Airports)
- Dropdowns feature smart formatting: **`{Full Airport Name} — {City/Place} [{IATA Code}]`**.
  - *Example:* `Indira Gandhi International Airport — Delhi [DEL]`
  - *Example:* `Chhatrapati Shivaji Maharaj International Airport — Mumbai [BOM]`
  - *Example:* `Dr. Babasaheb Ambedkar International Airport — Nagpur [NAG]`
- Supports instant fuzzy search by airport name, city, or 3-letter IATA code.

### 3. 🏢 Multi-Carrier Comparison & "All Airlines" Option
- Sidebar preferred carrier selectbox includes **`All Airlines (Compare All)`** alongside individual carriers (*IndiGo, Air India, Vistara, SpiceJet, AirAsia India, GO FIRST, Akasa Air*).
- When "Compare All" is selected, the platform calculates the full 6-carrier price spread (`₹ Min – ₹ Max`), highlights the best-value airline, and computes the route market average.
- Dual-mode multi-carrier matrix: Toggle between **Live Real-Time Online Tariffs** and **Supervised ML Model Benchmark**.
- Grouped Plotly comparison bar chart showing ML prediction vs. live market fare side-by-side for each carrier.

### 4. 🗺️ Geospatial Flight Corridor & Break-Journey Simulation
- Multi-layer Google Maps (Satellite Hybrid, Streets, Terrain) powered by Leaflet.js.
- Adaptive `fitBounds` with `maxZoom: 8` ensuring both airports are framed with proportional padding regardless of route distance.
- **Break-Journey / 1-Stop Support:** Routes exceeding 1,200 km or serving regional tier airports automatically render a **2-leg curved trajectory** with an amber pulsing stopover hub marker, "Leg 1" & "Leg 2" labels, and layover telemetry.
- Realistic aircraft animation with glowing neon flight arc.

### 5. 🧾 Itemized Bill of Airfare & Linear Breakdown
- Decomposes airfare into statutory and operating components:
  - 🔵 **Base Airfare:** Airline fleet operations, crew, and corporate margin (~45–55%).
  - 🟡 **Fuel & Dynamic Surcharge (YQ):** Aviation Turbine Fuel (ATF) and yield surge (~25–30%).
  - 🟢 **Airport Development Fees (UDF/PSF/ASF):** Terminal usage and DGCA aviation security fees.
  - 🟣 **Statutory GST:** 5% for Economy Class, 12% for Business Class.
- Custom animated linear HTML progress bars color-matched to each component, plus a combined stacked composition bar.

### 6. 📄 Universal 1-Page Executive PDF Quotation
- Built with ReportLab 5.x.
- Strictly formatted to fit exactly on a single standard A4 page.
- Clean INR currency formatting, flight itinerary specs, passenger carbon emissions ($CO_2$), itemized cost breakdown, and AI valuation advisory.

---

## 🏆 Model Performance Benchmark (300,153 Flights)

All models were evaluated using an 80/20 train-test partition on 300,153 validated Indian domestic flights:

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute % Error (MAPE) | $R^2$ Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | ₹4,549.50 | ₹6,727.83 | 46.81% | 0.9122 | 1.38s |
| **Ridge Regression** | ₹4,549.49 | ₹6,727.83 | 46.81% | 0.9122 | 0.73s |
| **Decision Tree Regressor** | ₹2,023.24 | ₹4,035.40 | 13.15% | 0.9684 | 2.26s |
| **HistGradientBoosting** | ₹2,208.53 | ₹3,773.38 | 17.01% | 0.9724 | 9.55s |
| **Random Forest Regressor (Best)** | **₹1,819.17** | **₹3,637.18** | **11.93%** | **0.9743** | 35.27s |

### 🔍 Key Pricing Insights & Feature Importance
1. **Cabin Class Disparity (86.1% Weight):** Business Class tickets trade at a structural multiple of **4× to 6×** over Economy, dominating variance in raw fare.
2. **Booking Window Non-Linear Surge (`days_left`):** Fares exhibit exponential price escalation within the final **7 days** before departure. Tickets purchased **20–35 days prior** remain stable at baseline promotional rates.
3. **Airline Brand Segmentation:** Full-service carriers (*Air India, Vistara*) trade at a premium tier, while low-cost carriers (*IndiGo, SpiceJet, AirAsia India*) aggressively compete on discount fare buckets.

---

## 📁 Repository Structure

```
airfare-price-estimation/
├── data/
│   ├── Clean_Dataset.csv          # 300,153 validated flight records
│   ├── economy.csv                # Raw Economy records
│   └── business.csv               # Raw Business records
├── src/
│   ├── airports_data.py           # 50+ airport database, GPS coordinates, Haversine routing
│   ├── pricing_engine.py          # Yield intelligence, multi-carrier matrix, bill decomposition
│   ├── realtime_pricing.py        # Live Google Flights API integration & market variance
│   ├── pdf_generator.py           # ReportLab 1-page executive PDF quotation generator
│   ├── data_preprocessing.py      # Feature engineering, ColumnTransformer, interaction terms
│   ├── train_models.py            # Supervised model training, evaluation & metrics export
│   └── evaluate.py                # Publication-grade regression diagnostic plots
├── models/
│   ├── best_model.joblib          # Serialized Random Forest pipeline (compressed, 97.43% R²)
│   ├── metrics.json               # Benchmark results across all 5 algorithms
│   ├── feature_importance.json    # Ranked feature weights (JSON)
│   └── feature_importance.csv     # Ranked feature weights (CSV)
├── plots/
│   ├── model_comparison.png       # R², MAE, RMSE publication comparison bar charts
│   ├── feature_importance.png     # Component pricing driver contribution chart
│   ├── predicted_vs_actual.png    # Scatter plot with ±10% linear corridor
│   ├── residuals_distribution.png # Residual error distribution with Gaussian bell curve
│   └── days_left_vs_price.png     # Booking window surge curve by cabin class
├── .vscode/
│   ├── launch.json                # 1-Click F5 debug & launch configurations
│   └── settings.json              # Python environment preferences
├── app.py                         # Enterprise Streamlit Web Dashboard
├── requirements.txt               # Project dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/KTejash/airfare-price-estimation.git
cd airfare-price-estimation
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 💻 Tech Stack & Libraries

- **Language:** Python 3.10+
- **Machine Learning & Preprocessing:** Scikit-Learn, NumPy, Pandas, SciPy, Joblib
- **Web Application & UI:** Streamlit, Streamlit Components (iFrame isolation)
- **Data Visualization:** Plotly Graph Objects, Plotly Express, Matplotlib, Seaborn
- **Geospatial & Mapping:** Leaflet.js, OpenStreetMap, Google Maps Satellite/Streets/Terrain tiles
- **Live Data Integration:** `fast-flights` (Protobuf-encoded Google Flights OTA integration)
- **Document Generation:** ReportLab 5.x (Executive PDF Quotations)

---

## 📜 Citation & Academic Reference

If using this project for academic reference, hackathon benchmarks, or coursework:

```bibtex
@misc{singh2026airfare,
  author = {Tejash Kumar Singh},
  title = {Enterprise Airfare Component Price Estimation and Yield Intelligence System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/KTejash/airfare-price-estimation}},
  note = {Smart India Hackathon 2026 (SIH26056), Supervised Learning (INT395)}
}
```

---

## 📄 License
This project is open-source and distributed under the [MIT License](LICENSE).
