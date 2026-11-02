# Airfare Component Price Estimation & Yield Intelligence System
**Smart India Hackathon (SIH 2026 — SIH26056)**  
**Course:** Supervised Learning (INT395) | Lovely Professional University  
**Topic S.No:** 17

---

## 🌟 What Makes This Implementation Stand Out
While standard student submissions only test 5–6 metro cities with basic regression, this project extends the research into a **production-grade enterprise aviation intelligence platform**:

1. **Nationwide Coverage (50+ Indian Airports & Cities):**
   - Supports 50+ commercial airports across all 28 States & UTs (Goa, Jaipur, Srinagar, Kochi, Leh, Guwahati, Amritsar, Lucknow, Port Blair, etc.).
   - Integrated with exact GPS coordinates, IATA codes (DEL, BOM, BLR, GOI, SXR, COK, etc.), and airport tier classifications (Metro, Tier-2, Tier-3).
2. **Great-Circle Geodesic Physics & Dynamic Routing:**
   - Computes great-circle distances via the Haversine formula, true compass bearing, realistic aircraft block duration (~650 km/h cruising speed), and layover recommendations.
3. **Multi-Carrier Benchmarking Matrix:**
   - Compares all 6 major Indian carriers (IndiGo, Air India, Vistara, SpiceJet, AirAsia India, GO FIRST) simultaneously on any chosen route and booking window.
4. **Interactive Geospatial Flight Path (PyDeck 3D Map):**
   - Renders 3D geodesic flight arcs and airport pins directly on an interactive map of India.
5. **Component Cost Breakdown (Itemized Airfare Bill):**
   - Decomposes the estimated airfare into true aviation fare components: Base Airfare, Fuel & Dynamic Surcharge (YQ), Airport Development Fees (UDF/PSF/ASF), and Statutory GST (5% Economy / 12% Business) with an interactive Plotly Donut Chart.
6. **Smart "Book Now vs Wait" Yield Management Advisor:**
   - Evaluates price acceleration and days left to advise consumers whether to buy immediately (avoiding 72-hour surges) or hold for early-bird discounts.
7. **Carbon Footprint ($CO_2$ Index):**
   - Calculates flight distance in km/miles and per-passenger carbon emissions ($kg\ CO_2$).

---

## 🏆 Model Performance Benchmark Summary (300,153 Records)

| Model | MAE (₹) | RMSE (₹) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | ₹4,549.50 | ₹6,727.83 | 46.81% | 0.9122 |
| **Ridge Regression** | ₹4,549.49 | ₹6,727.83 | 46.81% | 0.9122 |
| **Decision Tree** | ₹2,023.24 | ₹4,035.40 | 13.15% | 0.9684 |
| **HistGradientBoosting** | ₹2,208.53 | ₹3,773.38 | 17.01% | 0.9724 |
| **Random Forest (Best)** | **₹1,819.17** | **₹3,637.18** | **11.93%** | **0.9743** |

---

## 📁 Repository Structure
```
airfare-price-estimation/
├── data/
│   ├── Clean_Dataset.csv          # 300,153 validated flight records
│   ├── economy.csv                # Raw Economy records
│   └── business.csv               # Raw Business records
├── src/
│   ├── airports_data.py           # 50+ airport database, GPS coordinates, Haversine engine
│   ├── pricing_engine.py          # Yield intelligence, multi-carrier matrix, bill decomposition
│   ├── data_preprocessing.py      # Data ingestion, route interaction, ColumnTransformer
│   ├── train_models.py            # Supervised model training & permutation importance
│   └── evaluate.py                # Publication-quality diagnostic plots
├── models/
│   ├── best_model.joblib          # Serialized Random Forest pipeline (97.43% R²)
│   ├── metrics.json               # Benchmark results across all algorithms
│   └── feature_importance.csv     # Ranked feature weights
├── plots/
│   ├── model_comparison.png       # R², MAE, RMSE benchmark bar chart
│   ├── feature_importance.png     # Component pricing drivers bar chart
│   ├── predicted_vs_actual.png    # Scatter plot of actual vs predicted fares
│   ├── residuals_distribution.png # Residual error distribution
│   └── days_left_vs_price.png     # Booking window surge curve
├── .vscode/
│   ├── launch.json                # 1-Click F5 debug & launch configurations
│   └── settings.json              # Python environment preferences
├── app.py                         # Enterprise Streamlit Web Application
├── PROJECT_REPORT.md              # Complete academic CA1 submission report
└── requirements.txt               # Dependencies
```

---

## 🚀 How to Run in VS Code

1. Open folder in VS Code:
   `C:\Users\TEJASH KUMAR SINGH\.gemini\antigravity\scratch\airfare-price-estimation`
2. Open terminal (`Ctrl + ` `) and run:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser.
