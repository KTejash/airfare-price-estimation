"""
✈️ Enterprise Airfare Component Price Estimation & Yield Intelligence System
Smart India Hackathon (SIH 2026 — SIH26056) | Course: Supervised Learning (INT395)
Features:
- Medium-Dark Slate Aviation Theme with High-Contrast Typography
- User-Triggered "Estimate Airfare" Workflow (Input -> Button -> Results)
- Embedded Google Maps Satellite / Streets / Terrain Flight Path with Airplane Animation
- 50+ Indian Commercial Airports with Geodesic Routing
- Multi-Carrier Real-Time Tariff Benchmark
- Aviation Component Bill Breakdown & Yield Management Sensitivity Simulator
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
from pdf_generator import generate_flight_quotation_pdf

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

# ----------------- MEDIUM-DARK AVIATION THEME CSS -----------------
st.markdown("""
<style>
    /* Global medium-dark palette styling */
    .stApp {
        background-color: #1a2234;
        color: #f1f5f9;
    }
    
    /* Top title styling */
    h1, h2, h3, h4, h5 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #141c2b !important;
        border-right: 1px solid #2d3748;
    }
    
    /* Metric Card Values */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #60a5fa !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }
    
    /* Custom container borders */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #242f44 !important;
        border: 1px solid #3b4d6b !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    /* Tab buttons */
    button[data-baseweb="tab"] {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
        border-bottom-color: #3b82f6 !important;
    }
    
    /* Primary button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: 1px solid #60a5fa !important;
        padding: 0.65rem 1.25rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45);
    }
    
    /* Pill badges */
    .best-value-badge {
        background-color: #10b981;
        color: #ffffff !important;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .advisory-badge {
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
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

# ----------------- SESSION STATE FOR BUTTON TRIGGER -----------------
if "has_calculated" not in st.session_state:
    st.session_state.has_calculated = False

# ----------------- HEADER -----------------
col_hdr1, col_hdr2 = st.columns([3.8, 1.4])
with col_hdr1:
    st.title("✈️ Airfare Component Price Estimation & Yield Engine")
    st.markdown("**Smart India Hackathon (SIH 2026 — SIH26056)** • *Course: Supervised Learning (INT395)*")

with col_hdr2:
    with st.container(border=True):
        st.markdown("<div style='color: #60a5fa; font-weight:700; font-size:13px;'>🏛️ SIH 2026 Problem ID</div>", unsafe_allow_html=True)
        st.markdown("**Code:** `SIH26056` • **Accuracy:** `97.43% R²`")

st.markdown("---")

# ----------------- SIDEBAR PARAMETERS -----------------
st.sidebar.header("🛫 Flight Itinerary Configurator")
st.sidebar.caption("Configure your journey below, then click Estimate Airfare.")

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

# MAIN ACTION BUTTON IN SIDEBAR AND MAIN BODY
st.sidebar.markdown("---")
btn_sidebar = st.sidebar.button("🚀 Calculate & Show Airfare", type="primary", use_container_width=True)
if btn_sidebar:
    st.session_state.has_calculated = True

# ----------------- GOOGLE MAPS HTML GENERATOR -----------------
def generate_google_maps_html(src_info, dst_info, distance_km, duration_hrs):
    """
    Renders an interactive Google Maps powered Leaflet canvas with satellite/terrain/streets
    layer switcher, airport radar pins, and an animated flight trajectory.
    """
    lat1, lon1 = src_info['lat'], src_info['lon']
    lat2, lon2 = dst_info['lat'], dst_info['lon']
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    
    # Calculate fit bounds
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background: #1a2234; }}
            #map {{ width: 100%; height: 500px; border-radius: 10px; border: 1px solid #3b4d6b; }}
            .airport-label {{
                background: rgba(20, 28, 43, 0.88);
                color: #f8fafc;
                font-weight: 700;
                font-size: 12px;
                padding: 4px 8px;
                border-radius: 6px;
                border: 1px solid #60a5fa;
                box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            }}
            .leaflet-popup-content-wrapper {{
                background: #1e293b;
                color: #f8fafc;
                border-radius: 8px;
                border: 1px solid #475569;
            }}
            .leaflet-popup-tip {{ background: #1e293b; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            // Initialize map with neutral center
            var map = L.map('map', {{
                center: [{mid_lat}, {mid_lon}],
                zoom: 5,
                zoomControl: true
            }});

            // Google Maps Tile Layers
            var gStreets = L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}', {{
                maxZoom: 20,
                attribution: '© Google Maps'
            }});

            var gSatellite = L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}', {{
                maxZoom: 20,
                attribution: '© Google Maps Satellite'
            }});

            var gTerrain = L.tileLayer('https://mt1.google.com/vt/lyrs=p&x={{x}}&y={{y}}&z={{z}}', {{
                maxZoom: 20,
                attribution: '© Google Maps Terrain'
            }});

            // Add default Satellite layer for high visual appeal
            gSatellite.addTo(map);

            var baseLayers = {{
                "🛰️ Google Satellite (Hybrid)": gSatellite,
                "🗺️ Google Streets": gStreets,
                "⛰️ Google Terrain": gTerrain
            }};
            L.control.layers(baseLayers, null, {{ position: 'topright' }}).addTo(map);

            // Fit bounds with padding so both airports are clearly framed
            map.fitBounds([
                [{min_lat}, {min_lon}],
                [{max_lat}, {max_lon}]
            ], {{ padding: [60, 60] }});

            // Origin Marker (Cyan Radar Pulse)
            var originMarker = L.circleMarker([{lat1}, {lon1}], {{
                radius: 10,
                fillColor: '#38bdf8',
                color: '#ffffff',
                weight: 3,
                opacity: 1,
                fillOpacity: 0.9
            }}).addTo(map);
            originMarker.bindPopup("<b>🛫 ORIGIN AIRPORT</b><br>{src_info['name']}<br><b>Code:</b> {src_info['code']} • <b>Tier:</b> {src_info['tier']}").openPopup();

            // Destination Marker (Coral Red Pulse)
            var destMarker = L.circleMarker([{lat2}, {lon2}], {{
                radius: 10,
                fillColor: '#f43f5e',
                color: '#ffffff',
                weight: 3,
                opacity: 1,
                fillOpacity: 0.9
            }}).addTo(map);
            destMarker.bindPopup("<b>🛬 DESTINATION AIRPORT</b><br>{dst_info['name']}<br><b>Code:</b> {dst_info['code']} • <b>Tier:</b> {dst_info['tier']}");

            // Curved Parabolic Flight Path Generator
            function getCurvedPath(p1, p2, numPoints) {{
                var points = [];
                var dLat = p2[0] - p1[0];
                var dLon = p2[1] - p1[1];
                var dist = Math.sqrt(dLat * dLat + dLon * dLon);
                var curveFactor = 0.18; // curve deflection height

                for (var i = 0; i <= numPoints; i++) {{
                    var t = i / numPoints;
                    var lat = p1[0] + dLat * t;
                    var lon = p1[1] + dLon * t;
                    // Orthogonal displacement for arc
                    var offset = Math.sin(t * Math.PI) * dist * curveFactor;
                    var curvedLat = lat + (-dLon / dist) * offset;
                    var curvedLon = lon + (dLat / dist) * offset;
                    points.push([curvedLat, curvedLon]);
                }}
                return points;
            }}

            var flightArcPoints = getCurvedPath([{lat1}, {lon1}], [{lat2}, {lon2}], 50);

            // Glowing Outer Glow Line
            L.polyline(flightArcPoints, {{
                color: '#38bdf8',
                weight: 6,
                opacity: 0.4
            }}).addTo(map);

            // Core Flight Path Line
            var flightLine = L.polyline(flightArcPoints, {{
                color: '#fbbf24',
                weight: 3,
                opacity: 0.95,
                dashArray: '8, 8'
            }}).addTo(map);

            // Animated Airplane Icon traversing the arc
            var planeIcon = L.divIcon({{
                html: "<div style='font-size:22px; transform: rotate(45deg); filter: drop-shadow(0 0 6px #fbbf24);'>✈️</div>",
                className: '',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            }});

            var planeMarker = L.marker(flightArcPoints[0], {{ icon: planeIcon }}).addTo(map);

            var step = 0;
            function animatePlane() {{
                step = (step + 1) % flightArcPoints.length;
                planeMarker.setLatLng(flightArcPoints[step]);
                setTimeout(animatePlane, 100);
            }}
            animatePlane();
        </script>
    </body>
    </html>
    """
    return html_code

# ----------------- MAIN VIEW LOGIC -----------------
if not st.session_state.has_calculated:
    # PROMPT BANNER WHEN INPUTS HAVE NOT BEEN CALCULATED YET
    with st.container(border=True):
        st.markdown(f"""
        <div style="text-align: center; padding: 25px 15px;">
            <div style="font-size: 42px; margin-bottom: 8px;">✈️</div>
            <h2 style="color: #60a5fa; margin-bottom: 8px;">Ready to Analyze Your Airfare Route</h2>
            <p style="color: #cbd5e1; font-size: 16px; max-width: 650px; margin: 0 auto 20px auto;">
                Currently configured: <b style="color:#ffffff;">{origin_city} ({AIRPORTS[origin_city]['code']})</b> ➔ 
                <b style="color:#ffffff;">{dest_city} ({AIRPORTS[dest_city]['code']})</b> on 
                <b style="color:#ffffff;">{selected_airline}</b> ({flight_class} Class, {days_left} days prior).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
        with c_btn2:
            if st.button("🚀 Calculate & Show Airfare Breakdown", type="primary", use_container_width=True):
                st.session_state.has_calculated = True
                st.rerun()

else:
    # RECALCULATE / RESET ROW
    rcol1, rcol2 = st.columns([4, 1.2])
    with rcol1:
        st.caption(f"Showing results for: **{origin_city} ➔ {dest_city}** | **{selected_airline}** | **{flight_class}** | **{days_left} days left**")
    with rcol2:
        if st.button("🔄 Recalculate / Update", use_container_width=True):
            st.session_state.has_calculated = True
            st.rerun()

    # ----------------- MAIN NAVIGATION TABS -----------------
    tab_fare, tab_multi, tab_map, tab_benchmark, tab_insights = st.tabs([
        "🎯 Fare Prediction & Bill",
        "🏢 Multi-Airline Price Matrix",
        "🗺️ Live Google Maps Flight Path",
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
            m_col1, m_col2, m_col3 = st.columns([1.5, 1.4, 1.3])

            with m_col1:
                with st.container(border=True):
                    st.metric(
                        label=f"Predicted Airfare ({selected_airline} - {flight_class})",
                        value=f"₹ {fare_val:,.0f}",
                        delta=f"Booking Window: {days_left} days left"
                    )
                    st.caption(f"Estimated Range (±MAE): **₹ {max(1500, fare_val - 1819):,.0f} – ₹ {fare_val + 1819:,.0f}**")

            with m_col2:
                with st.container(border=True):
                    st.markdown(f"<span style='color: {advisory['color']}; font-weight:800; font-size:15px;'>{advisory['icon']} {advisory['status']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 13px; color: #cbd5e1; margin-top: 6px;'>{advisory['action']}</div>", unsafe_allow_html=True)
                    st.caption(f"**Expected Trend:** {advisory['expected_trend']}")

            with m_col3:
                with st.container(border=True):
                    st.markdown("<span style='color: #94a3b8; font-weight: 700; font-size: 12px;'>FLIGHT ROUTE SPECS</span>", unsafe_allow_html=True)
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
                        ("Base Airfare (Operations & Fleet Margin)", breakdown["Base Fare"], "#60a5fa"),
                        ("Fuel & Dynamic Demand Surcharge (YQ)", breakdown["Fuel & Dynamic Surcharge (YQ)"], "#fbbf24"),
                        ("Airport Development Fee (UDF/PSF/ASF)", breakdown["Airport Development Fee (UDF/PSF)"], "#34d399"),
                        ("Statutory Goods & Services Tax (GST)", breakdown["Goods & Services Tax (GST)"], "#c084fc"),
                    ]

                    for label, amt, col_code in breakdown_items:
                        b_pct = (amt / fare_val) * 100
                        st.markdown(f"<span style='color: {col_code}; font-weight:700;'>●</span> **{label}:** `₹ {amt:,.2f}` *({b_pct:.1f}%)*", unsafe_allow_html=True)
                        st.progress(min(1.0, max(0.0, b_pct / 100)))

                    st.markdown("---")
                    st.markdown(f"### Total Invoice Amount: <span style='color: #60a5fa;'>₹ {fare_val:,.2f}</span>", unsafe_allow_html=True)

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
                        hole=.52,
                        marker=dict(colors=['#60a5fa', '#fbbf24', '#34d399', '#c084fc']),
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
                color_discrete_sequence=["#60a5fa"]
            )
            fig_curve.add_vline(
                x=days_left, line_dash="dash", line_color="#f43f5e",
                annotation_text=f"Selected: Day {days_left}", annotation_font_color="#f43f5e"
            )
            fig_curve.update_xaxes(autorange="reversed", title="Days Before Departure (Decreasing towards flight)")
            fig_curve.update_yaxes(title="Estimated Airfare (₹)")
            fig_curve.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=370,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            # Download Itinerary Button (Executive PDF Report Only)
            st.markdown("---")
            pdf_bytes = generate_flight_quotation_pdf(
                origin_city=origin_city,
                dest_city=dest_city,
                src_info=AIRPORTS[origin_city],
                dst_info=AIRPORTS[dest_city],
                selected_airline=selected_airline,
                flight_class=flight_class,
                dep_time=dep_time,
                arr_time=arr_time,
                days_left=days_left,
                route_info=route_info,
                fare_val=fare_val,
                breakdown=breakdown,
                advisory=advisory
            )

            st.download_button(
                label="📄 Download Official Airfare Quotation (Executive PDF Report)",
                data=pdf_bytes,
                file_name=f"Flight_Quotation_{AIRPORTS[origin_city]['code']}_{AIRPORTS[dest_city]['code']}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            st.caption("💡 Official 1-Page PDF Quotation universally compatible with all phones, tablets, and computers. Contains verified flight specs, itemized cost breakdown, and AI valuation.")

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

        carrier_colors = {
            "IndiGo": "#38bdf8",
            "Air India": "#f43f5e",
            "Vistara": "#a855f7",
            "AirAsia India": "#fb923c",
            "SpiceJet": "#fbbf24",
            "GO FIRST": "#34d399"
        }

        cols_cards = st.columns(3)
        for idx, row in df_multi.iterrows():
            c_idx = idx % 3
            with cols_cards[c_idx]:
                is_cheapest = (row['Estimated_Fare'] == lowest_fare)
                bar_color = carrier_colors.get(row['Airline'], "#60a5fa")

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
                    st.markdown(f"<div style='font-size:24px; font-weight:800; color:#60a5fa; margin: 4px 0;'>₹ {row['Estimated_Fare']:,.0f}</div>", unsafe_allow_html=True)
                    st.markdown(f"⏱️ **Duration:** `{row['Duration_Hours']} hrs` • **Stops:** `{row['Stops']}`")

        st.markdown("---")
        st.markdown("#### 📋 Comprehensive Comparison Matrix")

        st.dataframe(
            df_multi.style.format({'Estimated_Fare': '₹{:,.0f}', 'Duration_Hours': '{:.1f} hrs'}),
            use_container_width=True
        )

    # ==================== TAB 3: LIVE GOOGLE MAPS FLIGHT PATH ====================
    with tab_map:
        st.subheader(f"🗺️ Live Google Maps Flight Path: {origin_city} ➔ {dest_city}")
        st.caption(f"Connecting **{AIRPORTS[origin_city]['name']}** ({AIRPORTS[origin_city]['code']}) to **{AIRPORTS[dest_city]['name']}** ({AIRPORTS[dest_city]['code']})")

        # Top Google Maps Direct Link Button
        src_lat, src_lon = AIRPORTS[origin_city]['lat'], AIRPORTS[origin_city]['lon']
        dst_lat, dst_lon = AIRPORTS[dest_city]['lat'], AIRPORTS[dest_city]['lon']
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={src_lat},{src_lon}&destination={dst_lat},{dst_lon}"

        gm_col1, gm_col2 = st.columns([3.5, 1.5])
        with gm_col1:
            st.markdown("💡 *Use the top-right layer switch on the map to toggle between **Google Satellite (Hybrid)**, **Google Streets**, and **Google Terrain**.*")
        with gm_col2:
            st.link_button("🌐 Open Live in Google Maps", gmaps_url, use_container_width=True)

        # Render Embedded Google Maps Canvas
        map_html = generate_google_maps_html(
            AIRPORTS[origin_city], AIRPORTS[dest_city], route_info['distance_km'], route_info['est_duration']
        )
        components.html(map_html, height=520, scrolling=False)

        # Flight telemetries
        st.markdown("---")
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
                name='R² Score', marker_color='#60a5fa',
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
