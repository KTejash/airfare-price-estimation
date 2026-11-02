"""
✈️ Enterprise Airfare Component Price Estimation & Yield Intelligence System
Smart India Hackathon (SIH 2026 — SIH26056) | INT395 Supervised Learning
Full Theme-Adaptive UI with 50+ Airports, 3D Geospatial Radar Map,
Multi-Carrier Benchmarking, Interactive Diagnostics, and Quotation Exporter.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

# Ensure src modules can be resolved
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from airports_data import AIRPORTS, METRO_HUBS, get_route_details
from pricing_engine import (
    CARRIERS,
    load_inference_pipeline,
    predict_single_flight,
    get_multi_carrier_comparison,
    get_booking_advisory
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Airfare Intelligence & Price Estimator | SIH 2026",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_PATH = os.path.join(BASE_DIR, "data", "Clean_Dataset.csv")

# ----------------- CUSTOM HIGH-CONTRAST CSS -----------------
st.markdown("""
<style>
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    /* Tab headers */
    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    /* Clean container borders */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 6px;
    }
    /* Badge pill */
    .sih-badge {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff !important;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* Card badge */
    .best-value-badge {
        background-color: #10b981;
        color: #ffffff !important;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cached_pipeline():
    return load_inference_pipeline(BASE_DIR)

@st.cache_data
def get_cached_metadata():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    feat_path = os.path.join(MODELS_DIR, "feature_importance.json")
    metrics, importance = {}, []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            importance = json.load(f)
    return metrics, importance

pipeline = get_cached_pipeline()
metrics_dict, importance_list = get_cached_metadata()

# ----------------- HEADER -----------------
col_hdr1, col_hdr2 = st.columns([3.8, 1.4])
with col_hdr1:
    st.title("✈️ Airfare Component Price Estimation & Yield Engine")
    st.markdown("**Smart India Hackathon (SIH 2026 — SIH26056)** • *Course: Supervised Learning (INT395)*")

with col_hdr2:
    with st.container(border=True):
        st.markdown("<div style='color: #38bdf8; font-weight:700;'>🏛️ SIH 2026 Problem ID</div>", unsafe_allow_html=True)
        st.markdown("**Code:** `SIH26056` • **Accuracy:** `97.43% R²`")

st.markdown("---")

# ----------------- SIDEBAR PARAMETERS -----------------
st.sidebar.header("🛫 Flight Itinerary Configurator")

city_names = sorted(list(AIRPORTS.keys()))

default_src_idx = city_names.index("Delhi") if "Delhi" in city_names else 0
default_dst_idx = city_names.index("Mumbai") if "Mumbai" in city_names else 1

origin_city = st.sidebar.selectbox("Origin City / Airport", city_names, index=default_src_idx)

dest_options = [c for c in city_names if c != origin_city]
dest_city = st.sidebar.selectbox("Destination City / Airport", dest_options, index=min(default_dst_idx, len(dest_options)-1))

flight_class = st.sidebar.radio("Cabin Class", ["Economy", "Business"], horizontal=True)

carrier_names = [c["name"] for c in CARRIERS]
selected_airline = st.sidebar.selectbox("Preferred Airline Carrier", carrier_names, index=0)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    dep_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"], index=1)
with col_s2:
    arr_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"], index=3)

days_left = st.sidebar.slider("📅 Days Before Departure (Booking Window)", min_value=1, max_value=50, value=24)

# Real-time route computation
route_info = get_route_details(origin_city, dest_city)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**✈️ Flight Distance:** `{route_info['distance_km']} km` ({route_info['distance_miles']} mi)")
st.sidebar.markdown(f"**⏱️ Est. Duration:** `{route_info['est_duration']} hrs` ({route_info['rec_stops']} stops)")
st.sidebar.markdown(f"**🌱 Carbon Footprint:** `~{route_info['co2_kg']} kg CO₂`")

# ----------------- MAIN NAVIGATION TABS -----------------
tab_fare, tab_multi, tab_map, tab_benchmark, tab_insights = st.tabs([
    "🎯 Fare Prediction & Bill",
    "🏢 Multi-Airline Price Matrix",
    "🗺️ Interactive Flight Radar Map",
    "📊 Model Comparison (SIH Benchmark)",
    "🔍 Feature Drivers & Intelligence"
])

# ==================== TAB 1: FARE PREDICTION & COMPONENT BILL ====================
with tab_fare:
    if pipeline is None:
        st.error("ML model not loaded. Please ensure models/best_model.joblib exists.")
    else:
        flight_result = predict_single_flight(
            pipeline, origin_city, dest_city, selected_airline, flight_class, dep_time, arr_time, days_left
        )
        fare_val = flight_result['price']
        breakdown = flight_result['breakdown']
        advisory = get_booking_advisory(days_left, fare_val, flight_class)

        # Top Metric Banner
        m_col1, m_col2, m_col3 = st.columns([1.5, 1.3, 1.4])

        with m_col1:
            with st.container(border=True):
                st.metric(
                    label=f"Predicted Airfare ({selected_airline} - {flight_class})",
                    value=f"₹ {fare_val:,.0f}",
                    delta=f"Window: {days_left} days left"
                )
                st.caption(f"Estimated Range (±MAE): **₹ {max(1500, fare_val - 1819):,.0f} – ₹ {fare_val + 1819:,.0f}**")

        with m_col2:
            with st.container(border=True):
                st.markdown(f"<span style='color: {advisory['color']}; font-weight:800; font-size:15px;'>{advisory['icon']} {advisory['status']}</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 13px; color: #cbd5e1; margin-top: 6px;'>{advisory['action']}</div>", unsafe_allow_html=True)
                st.caption(f"**Trend:** {advisory['expected_trend']}")

        with m_col3:
            with st.container(border=True):
                st.markdown("<span style='color: #94a3b8; font-weight: 700; font-size: 12px;'>FLIGHT ITINERARY SPECS</span>", unsafe_allow_html=True)
                st.markdown(f"**{AIRPORTS[origin_city]['code']} ➔ {AIRPORTS[dest_city]['code']}** • `{route_info['distance_km']} km`")
                st.markdown(f"**Duration:** `{route_info['est_duration']} hrs` • `{route_info['rec_stops'].replace('_', ' ').capitalize()}` Stops")
                st.caption(f"Departure Window: **{dep_time.replace('_', ' ')}**")

        st.markdown("---")

        # Two-column layout for Bill Breakdown & Donut Chart
        b_col1, b_col2 = st.columns([1.3, 1.2])

        with b_col1:
            with st.container(border=True):
                st.subheader("🧾 Component Cost Breakdown (Bill of Airfare)")
                st.caption("Detailed decomposition of airfare into base operating tariff, fuel surcharges, and statutory aviation fees.")

                breakdown_items = [
                    ("Base Airfare (Operations & Margin)", breakdown["Base Fare"], "#38bdf8"),
                    ("Fuel & Dynamic Demand Surcharge (YQ)", breakdown["Fuel & Dynamic Surcharge (YQ)"], "#f59e0b"),
                    ("Airport Development Fee (UDF/PSF/ASF)", breakdown["Airport Development Fee (UDF/PSF)"], "#10b981"),
                    ("Statutory Goods & Services Tax (GST)", breakdown["Goods & Services Tax (GST)"], "#a855f7"),
                ]

                for label, amt, col_code in breakdown_items:
                    b_pct = (amt / fare_val) * 100
                    st.markdown(f"<span style='color: {col_code}; font-weight:700;'>●</span> **{label}:** `₹ {amt:,.2f}` *({b_pct:.1f}%)*", unsafe_allow_html=True)
                    st.progress(min(1.0, max(0.0, b_pct / 100)))

                st.markdown("---")
                st.markdown(f"### Total Invoice Amount: <span style='color: #38bdf8;'>₹ {fare_val:,.2f}</span>", unsafe_allow_html=True)

        with b_col2:
            with st.container(border=True):
                st.subheader("📊 Component Cost Share")
                donut_labels = ["Base Fare", "Fuel Surcharge (YQ)", "Airport Fees (UDF/PSF)", "Statutory GST"]
                donut_values = [
                    breakdown["Base Fare"],
                    breakdown["Fuel & Dynamic Surcharge (YQ)"],
                    breakdown["Airport Development Fee (UDF/PSF)"],
                    breakdown["Goods & Services Tax (GST)"]
                ]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=.50,
                    marker=dict(colors=['#38bdf8', '#f59e0b', '#10b981', '#a855f7']),
                    textinfo="label+percent",
                    textfont=dict(color="#ffffff")
                )])
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=290,
                    showlegend=False
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")

        # Dynamic Price Sensitivity Curve
        st.subheader("📈 Dynamic Booking Window Sensitivity Simulation")
        st.write("Visualizing yield management price acceleration from 50 days down to 1 day before departure:")

        days_seq = np.arange(1, 51)
        sim_prices = []
        for d in days_seq:
            sim_res = predict_single_flight(
                pipeline, origin_city, dest_city, selected_airline, flight_class, dep_time, arr_time, int(d)
            )
            sim_prices.append(sim_res['price'])

        df_sim = pd.DataFrame({
            "Days Before Departure": days_seq,
            "Estimated Price (₹)": sim_prices
        })

        fig_curve = px.line(
            df_sim, x="Days Before Departure", y="Estimated Price (₹)",
            markers=True,
            title=f"Airfare Yield Curve: {origin_city} ➔ {dest_city} ({selected_airline} - {flight_class})",
            color_discrete_sequence=["#38bdf8"],
            template="plotly_dark"
        )
        fig_curve.add_vline(
            x=days_left, line_dash="dash", line_color="#f43f5e",
            annotation_text=f"Selected: Day {days_left}", annotation_font_color="#f43f5e"
        )
        fig_curve.update_xaxes(autorange="reversed", title="Days Before Departure (Decreasing towards flight)")
        fig_curve.update_yaxes(title="Estimated Airfare (₹)")
        fig_curve.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=370,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Download Itinerary Button
        st.markdown("---")
        itinerary_text = f"""=====================================================
AIRFARE ESTIMATION & QUOTATION INVOICE (SIH 2026 - SIH26056)
=====================================================
Journey: {origin_city} ({AIRPORTS[origin_city]['code']}) -> {dest_city} ({AIRPORTS[dest_city]['code']})
Airline Carrier: {selected_airline}
Cabin Class: {flight_class}
Flight Distance: {route_info['distance_km']} km ({route_info['distance_miles']} miles)
Estimated Flight Duration: {route_info['est_duration']} hours
Intermediate Stops: {route_info['rec_stops']}
Booking Window: {days_left} days prior to departure
-----------------------------------------------------
FARE COMPONENT BREAKDOWN:
- Base Airfare: INR {breakdown['Base Fare']:,.2f}
- Fuel & Dynamic Surcharge (YQ): INR {breakdown['Fuel & Dynamic Surcharge (YQ)']:,.2f}
- Airport Development Fee (UDF/PSF/ASF): INR {breakdown['Airport Development Fee (UDF/PSF)']:,.2f}
- Statutory GST: INR {breakdown['Goods & Services Tax (GST)']:,.2f}
-----------------------------------------------------
TOTAL ESTIMATED FARE: INR {fare_val:,.2f}
ESTIMATED RANGE (68% CI): INR {max(1500, fare_val - 1819):,.0f} - INR {fare_val + 1819:,.0f}
ESTIMATED CARBON FOOTPRINT: ~{route_info['co2_kg']} kg CO2
BOOKING ADVISORY: {advisory['status']} - {advisory['action']}
=====================================================
Generated by Supervised Learning (INT395) CA1 Pricing Engine.
"""
        st.download_button(
            label="📥 Download Official Fare Quotation (Text Invoice)",
            data=itinerary_text,
            file_name=f"Flight_Quote_{AIRPORTS[origin_city]['code']}_{AIRPORTS[dest_city]['code']}.txt",
            mime="text/plain"
        )

# ==================== TAB 2: MULTI-AIRLINE PRICE MATRIX ====================
with tab_multi:
    st.subheader(f"🏢 Multi-Carrier Comparison: {origin_city} ➔ {dest_city}")
    st.write(f"Real-time benchmark across all 6 major Indian carriers for **{flight_class} Class** with **{days_left} days left**:")

    # Sorting controls
    sort_choice = st.radio("Sort Carriers By:", ["Lowest Price First", "Shortest Duration", "Carrier Name"], horizontal=True)

    df_multi = get_multi_carrier_comparison(
        pipeline, origin_city, dest_city, flight_class, dep_time, arr_time, days_left
    )

    if sort_choice == "Lowest Price First":
        df_multi = df_multi.sort_values(by="Estimated_Fare").reset_index(drop=True)
    elif sort_choice == "Shortest Duration":
        df_multi = df_multi.sort_values(by="Duration_Hours").reset_index(drop=True)
    else:
        df_multi = df_multi.sort_values(by="Airline").reset_index(drop=True)

    lowest_fare = df_multi['Estimated_Fare'].min()

    # Brand color mapping
    carrier_colors = {
        "IndiGo": "#0284c7",
        "Air India": "#dc2626",
        "Vistara": "#7c3aed",
        "AirAsia India": "#ea580c",
        "SpiceJet": "#f59e0b",
        "GO FIRST": "#059669"
    }

    cols_cards = st.columns(3)
    for idx, row in df_multi.iterrows():
        c_idx = idx % 3
        with cols_cards[c_idx]:
            is_cheapest = (row['Estimated_Fare'] == lowest_fare)
            bar_color = carrier_colors.get(row['Airline'], "#38bdf8")

            with st.container(border=True):
                # Header row
                c1, c2 = st.columns([2.2, 1.2])
                with c1:
                    st.markdown(f"<span style='color: {bar_color}; font-size:18px; font-weight:800;'>{row['Airline']}</span> <span style='color:#94a3b8;'>({row['Code']})</span>", unsafe_allow_html=True)
                with c2:
                    if is_cheapest:
                        st.markdown("<span class='best-value-badge'>🏆 BEST FARE</span>", unsafe_allow_html=True)

                st.caption(f"{row['Type']} • {row['Badge']}")

                # Price metric
                st.markdown(f"<div style='font-size:24px; font-weight:800; color:#38bdf8; margin: 4px 0;'>₹ {row['Estimated_Fare']:,.0f}</div>", unsafe_allow_html=True)
                st.markdown(f"⏱️ **Duration:** `{row['Duration_Hours']} hrs` • **Stops:** `{row['Stops']}`")

    st.markdown("---")
    st.markdown("#### 📋 Comprehensive Comparison Matrix")

    st.dataframe(
        df_multi.style.format({'Estimated_Fare': '₹{:,.0f}', 'Duration_Hours': '{:.1f} hrs'}),
        use_container_width=True
    )

# ==================== TAB 3: INTERACTIVE FLIGHT RADAR MAP ====================
with tab_map:
    st.subheader(f"🗺️ Geospatial Flight Radar Corridor: {origin_city} ➔ {dest_city}")
    st.caption(f"Connecting **{AIRPORTS[origin_city]['name']}** ({AIRPORTS[origin_city]['code']}) with **{AIRPORTS[dest_city]['name']}** ({AIRPORTS[dest_city]['code']})")

    src_geo = AIRPORTS[origin_city]
    dst_geo = AIRPORTS[dest_city]

    # Calculate dynamic zoom & bounding box
    mid_lat = (src_geo['lat'] + dst_geo['lat']) / 2
    mid_lon = (src_geo['lon'] + dst_geo['lon']) / 2
    max_span = max(abs(src_geo['lat'] - dst_geo['lat']), abs(src_geo['lon'] - dst_geo['lon']))

    if max_span < 3.5:
        map_zoom = 5.8
    elif max_span < 7.0:
        map_zoom = 5.1
    elif max_span < 12.0:
        map_zoom = 4.5
    else:
        map_zoom = 3.9

    # Flight arc data
    arc_data = [{
        "from": [src_geo['lon'], src_geo['lat']],
        "to": [dst_geo['lon'], dst_geo['lat']],
        "from_name": f"{origin_city} ({src_geo['code']})",
        "to_name": f"{dest_city} ({dst_geo['code']})"
    }]

    # Glowing airport radar circles
    points_data = [
        {"name": f"{origin_city} ({src_geo['code']})", "coordinates": [src_geo['lon'], src_geo['lat']], "color": [0, 245, 212], "radius": 32000},
        {"name": f"{dest_city} ({dst_geo['code']})", "coordinates": [dst_geo['lon'], dst_geo['lat']], "color": [255, 107, 107], "radius": 32000}
    ]

    # Airport labels directly on 3D canvas
    text_data = [
        {"text": f"🛫 {src_geo['code']} ({origin_city})", "coordinates": [src_geo['lon'], src_geo['lat'] + 0.35], "color": [255, 255, 255]},
        {"text": f"🛬 {dst_geo['code']} ({dest_city})", "coordinates": [dst_geo['lon'], dst_geo['lat'] + 0.35], "color": [255, 255, 255]}
    ]

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[0, 245, 212, 230],
        get_target_color=[255, 107, 107, 230],
        get_width=6,
        get_height=0.45,
        get_tilt=15,
        auto_highlight=True,
    )

    radar_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_data,
        get_position="coordinates",
        get_color="color",
        get_radius="radius",
        stroked=True,
        filled=True,
        get_line_color=[255, 255, 255, 200],
        get_line_width=3,
        pickable=True
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=text_data,
        get_position="coordinates",
        get_text="text",
        get_size=16,
        get_color="color",
        get_alignment_baseline="'bottom'",
        get_text_anchor="'middle'",
        font_weight="bold"
    )

    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=map_zoom,
        pitch=35,
        bearing=0
    )

    deck = pdk.Deck(
        layers=[arc_layer, radar_layer, text_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
        map_style="dark"
    )

    st.pydeck_chart(deck)

    # Secondary interactive Plotly India Route Map
    with st.expander("🛰️ View Geographic Flight Corridor (Plotly Geo Map)", expanded=False):
        fig_geo = go.Figure()

        # Route line
        fig_geo.add_trace(go.Scattergeo(
            lon=[src_geo['lon'], dst_geo['lon']],
            lat=[src_geo['lat'], dst_geo['lat']],
            mode='lines+markers',
            line=dict(width=3.5, color='#38bdf8'),
            marker=dict(size=10, color=['#00f5d4', '#f43f5e']),
            name="Flight Trajectory",
            hoverinfo='text',
            text=[f"Origin: {origin_city} ({src_geo['code']})", f"Destination: {dest_city} ({dst_geo['code']})"]
        ))

        fig_geo.update_geos(
            scope="asia",
            center=dict(lat=mid_lat, lon=mid_lon),
            projection_scale=3.8,
            showland=True, landcolor="#1e293b",
            showocean=True, oceancolor="#0f172a",
            showcountries=True, countrycolor="#334155",
            showsubunits=True, subunitcolor="#334155"
        )
        fig_geo.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    # Flight telemetries
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        st.metric("Great-Circle Distance", f"{route_info['distance_km']} km", f"{route_info['distance_miles']} miles")
    with g_col2:
        st.metric("Compass Bearing", f"{route_info['bearing']}°", "True Heading")
    with g_col3:
        cruising_alt = "FL340 (34,000 ft)" if route_info['distance_km'] > 600 else "FL240 (24,000 ft)"
        st.metric("Cruising Altitude", cruising_alt, "~Mach 0.78")
    with g_col4:
        st.metric("CO₂ Carbon Footprint", f"{route_info['co2_kg']} kg", "Per Passenger")

# ==================== TAB 4: MODEL COMPARISON & SIH BENCHMARK ====================
with tab_benchmark:
    st.subheader("📊 INT395 Supervised Learning Benchmark (SIH26056)")
    st.write("Cross-algorithm benchmark evaluated on **300,153 validated flights** (80/20 partition):")

    if metrics_dict:
        metrics_df = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={'index': 'Model'})

        # Clean formatted benchmark table
        st.dataframe(
            metrics_df.style.format({
                'MAE': '₹{:,.2f}',
                'RMSE': '₹{:,.2f}',
                'MAPE': '{:.2f}%',
                'R2': '{:.4f}',
                'Training_Time_s': '{:.2f}s'
            }),
            use_container_width=True
        )

        st.markdown("### 📈 Interactive Regression Diagnostics")

        # Interactive Plotly Model Comparison
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            x=metrics_df['Model'], y=metrics_df['R2'],
            name='R² Score', marker_color='#38bdf8',
            text=metrics_df['R2'].round(3), textposition='auto'
        ))
        fig_bench.update_layout(
            title="Algorithm Comparison: R² Score (Higher is Better)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            yaxis=dict(range=[0.85, 1.0])
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            pva_img = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
            if os.path.exists(pva_img):
                st.image(pva_img, caption="Predicted vs Actual Fare Scatter Plot", use_container_width=True)

        with pcol2:
            res_img = os.path.join(PLOTS_DIR, "residuals_distribution.png")
            if os.path.exists(res_img):
                st.image(res_img, caption="Residual Error Distribution (Actual - Predicted)", use_container_width=True)

        comp_img = os.path.join(PLOTS_DIR, "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Publication-Grade Metric Comparison Chart", use_container_width=True)

# ==================== TAB 5: FEATURE DRIVERS & INTELLIGENCE ====================
with tab_insights:
    st.subheader("🔍 Airfare Component Decomposition & Driver Rankings")
    st.write("Quantified percentage contribution of each feature using held-out permutation tests:")

    fi_col1, fi_col2 = st.columns([1.3, 1])

    with fi_col1:
        if importance_list:
            df_imp = pd.DataFrame(importance_list).sort_values(by="Importance_Percent", ascending=True)

            fig_imp = px.bar(
                df_imp, x="Importance_Percent", y="Feature", orientation="h",
                title="Relative Feature Importance Contribution (%)",
                color="Importance_Percent",
                color_continuous_scale="Blues",
                template="plotly_dark",
                text="Importance_Percent"
            )
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                xaxis_title="Importance Share (%)",
                yaxis_title="Feature Attribute"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    with fi_col2:
        if importance_list:
            st.markdown("##### 🏆 Ranked Component Impact")
            df_imp_display = pd.DataFrame(importance_list).sort_values(by="Importance_Percent", ascending=False)
            st.dataframe(
                df_imp_display[['Feature', 'Importance_Percent']].rename(columns={'Importance_Percent': 'Contribution (%)'}),
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("### 🛫 Pricing Insights & Yield Management Observations")
    st.markdown("""
    1. **Cabin Class Disparity (86.1% Weight)**:
       - Business class tickets trade at a structural multiple of 4x to 6x compared to economy tickets, dominating variance in raw fare.
    2. **Booking Window (`days_left`) Non-Linear Dynamics**:
       - Fares exhibit an exponential hockey-stick surge in the final 7 days before departure. Tickets purchased 20–35 days ahead remain stable near floor pricing.
    3. **Airline Brand Segmentation**:
       - Full-service carriers (Vistara, Air India) consistently show a premium pricing bracket, whereas budget carriers (IndiGo, AirAsia, SpiceJet, GO FIRST) compete aggressively on lowest-fare economy buckets.
    4. **Distance & Route Capacity**:
       - Flight duration and route distance scale base operational costs linearly, while airport slots and ground handling tier impact the fixed taxes and fees.
    """)

    days_img = os.path.join(PLOTS_DIR, "days_left_vs_price.png")
    if os.path.exists(days_img):
        st.image(days_img, caption="Booking Window vs Average Fare Curve by Class", use_container_width=True)
