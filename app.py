"""
✈️ Enterprise Airfare Component Price Estimation & Yield Intelligence System
Smart India Hackathon (SIH 2026 — SIH26056) | Course: Supervised Learning (INT395)
Features:
- Medium-Dark Slate Aviation Theme with High-Contrast Typography
- User-Triggered "Estimate Airfare" Workflow (Input -> Button -> Results)
- Embedded Google Maps Satellite / Streets / Terrain Flight Path with Airplane Animation
- 50+ Indian Commercial Airports with Geodesic Routing
- Multi-Carrier Real-Time Tariff Benchmark (rendered as pure HTML grid cards)
- Aviation Component Bill Breakdown & Yield Management Sensitivity Simulator
"""

import os
import sys
import json
from datetime import datetime, timedelta
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
from realtime_pricing import fetch_realtime_flights, compute_market_variance, get_selected_carrier_live_info

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
    /* Global medium-dark palette */
    .stApp {
        background-color: #111827;
        color: #f1f5f9;
    }

    /* Typography */
    h1, h2, h3, h4, h5 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    p, li, span, label {
        color: #cbd5e1 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #34d399 !important;
        font-size: 0.85rem !important;
    }

    /* Container borders */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        padding: 10px 14px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Primary buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(14,165,233,0.4);
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(14,165,233,0.55);
    }

    /* Secondary buttons */
    div.stButton > button:not([kind="primary"]) {
        background: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Download button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 12px rgba(5,150,105,0.4);
    }

    /* DataFrame table */
    [data-testid="stDataFrame"] {
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Progress bars */
    div[data-testid="stProgressBar"] > div {
        border-radius: 6px !important;
    }

    /* Dividers */
    hr { border-color: #1e3a5f !important; }

    /* Caption text */
    .stCaption, small {
        color: #64748b !important;
        font-size: 0.82rem !important;
    }

    /* Input widgets */
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stRadio"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
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

# ----------------- SESSION STATE -----------------
if "has_calculated" not in st.session_state:
    st.session_state.has_calculated = False

# ----------------- HEADER -----------------
st.title("✈️ Airfare Component Price Estimation & Yield Engine")
st.markdown(
    "<span style='color:#64748b; font-size:13px;'>"
    "Smart India Hackathon (SIH 2026 — SIH26056) • Course: Supervised Learning (INT395)"
    "</span>",
    unsafe_allow_html=True
)

st.markdown("---")

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("""
<div style="background:linear-gradient(135deg,#0ea5e9,#2563eb);
            border-radius:10px; padding:12px 14px; margin-bottom:12px; text-align:center;">
    <div style="color:#fff; font-weight:800; font-size:15px;">🛫 Flight Configurator</div>
    <div style="color:#bae6fd; font-size:11px; margin-top:3px;">
        Set route, class & booking window
    </div>
</div>
""", unsafe_allow_html=True)

def format_airport_label(city_key: str) -> str:
    ap = AIRPORTS[city_key]
    return f"{ap['name']} — {city_key} [{ap['code']}]"

# Smart airport display list sorted by city
airport_options = sorted([format_airport_label(c) for c in AIRPORTS.keys()])
label_to_city = {format_airport_label(c): c for c in AIRPORTS.keys()}

default_src_lbl = format_airport_label("Delhi") if "Delhi" in AIRPORTS else airport_options[0]
default_dst_lbl = format_airport_label("Mumbai") if "Mumbai" in AIRPORTS else airport_options[1]

origin_selection = st.sidebar.selectbox(
    "🛫 Origin Airport (Name — City [IATA])",
    airport_options,
    index=airport_options.index(default_src_lbl)
)
origin_city = label_to_city[origin_selection]

dest_airport_options = [lbl for lbl in airport_options if label_to_city[lbl] != origin_city]
dest_default_idx = dest_airport_options.index(default_dst_lbl) if default_dst_lbl in dest_airport_options else 0

dest_selection = st.sidebar.selectbox(
    "🛬 Destination Airport (Name — City [IATA])",
    dest_airport_options,
    index=dest_default_idx
)
dest_city = label_to_city[dest_selection]

flight_class = st.sidebar.radio("💺 Cabin Class", ["Economy", "Business"], horizontal=True)
carrier_names = ["All Airlines (Compare All)"] + [c["name"] for c in CARRIERS]
selected_airline = st.sidebar.selectbox("✈️ Preferred Carrier", carrier_names, index=0)
is_compare_all = (selected_airline == "All Airlines (Compare All)")

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    dep_time = st.selectbox("🕗 Departure", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"], index=1)
with col_s2:
    arr_time = st.selectbox("🕖 Arrival", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"], index=3)

days_left = st.sidebar.slider("📅 Days Before Departure", min_value=1, max_value=50, value=24)

# Real-time route computation
route_info = get_route_details(origin_city, dest_city)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background:#1e293b; border-radius:8px; padding:10px 12px; border-left:3px solid #0ea5e9;">
    <div style="color:#38bdf8; font-weight:700; font-size:12px; margin-bottom:6px;">📡 ROUTE TELEMETRY</div>
    <div style="color:#e2e8f0; font-size:13px; margin-bottom:3px;">✈️ Distance: <b>{route_info['distance_km']} km</b> ({route_info['distance_miles']} mi)</div>
    <div style="color:#e2e8f0; font-size:13px; margin-bottom:3px;">⏱️ Duration: <b>{route_info['est_duration']} hrs</b></div>
    <div style="color:#34d399; font-size:12px;">🌱 CO₂: ~{route_info['co2_kg']} kg/passenger</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
btn_sidebar = st.sidebar.button("🚀 Calculate & Estimate Airfare", type="primary", use_container_width=True)
if btn_sidebar:
    st.session_state.has_calculated = True


# ─────────────────────────────────────────────────────────────
#  ENHANCED FLIGHT SIMULATION MAP
# ─────────────────────────────────────────────────────────────
def generate_google_maps_html(src_info, dst_info, distance_km, duration_hrs,
                               stopover_info=None):
    lat1, lon1 = src_info["lat"], src_info["lon"]
    lat2, lon2 = dst_info["lat"], dst_info["lon"]

    # Tight adaptive bounding box
    all_lats = [lat1, lat2]
    all_lons = [lon1, lon2]
    has_stop = stopover_info is not None
    if has_stop:
        all_lats.append(stopover_info["lat"])
        all_lons.append(stopover_info["lon"])

    span_lat = max(all_lats) - min(all_lats)
    span_lon = max(all_lons) - min(all_lons)
    pad_lat  = max(1.0, span_lat * 0.25)
    pad_lon  = max(1.0, span_lon * 0.25)
    min_lat  = min(all_lats) - pad_lat
    max_lat  = max(all_lats) + pad_lat
    min_lon  = min(all_lons) - pad_lon
    max_lon  = max(all_lons) + pad_lon

    if has_stop:
        s = stopover_info
        slat = s["lat"]; slon = s["lon"]
        scode = s.get("code", "HUB")
        sname = s.get("name", "Connecting Hub")
        stier = s.get("tier", "Hub")
        stop_js = f"""
  // 1-STOP FLIGHT: linear flight corridors via layover hub
  var leg1 = linearPath([{lat1},{lon1}],[{slat},{slon}],60);
  var leg2 = linearPath([{slat},{slon}],[{lat2},{lon2}],60);
  var arc  = leg1.concat(leg2);
  [leg1,leg2].forEach(function(leg){{
    L.polyline(leg,{{color:'#38bdf8',weight:14,opacity:0.08}}).addTo(map);
    L.polyline(leg,{{color:'#38bdf8',weight:6, opacity:0.25}}).addTo(map);
    L.polyline(leg,{{color:'#fbbf24',weight:2.5,opacity:1.0,dashArray:'10 6'}}).addTo(map);
  }});
  var stopH=`<div style="position:relative;width:36px;height:36px;">
    <div class="pulse-ring" style="background:#f59e0b;opacity:0.5;animation-delay:.2s;"></div>
    <div class="pulse-ring" style="background:#f59e0b;opacity:0.3;animation-delay:.9s;"></div>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
      width:18px;height:18px;background:#f59e0b;border-radius:50%;
      border:3px solid #fff;box-shadow:0 0 14px #f59e0b;"></div></div>`;
  L.marker([{slat},{slon}],{{icon:L.divIcon({{html:stopH,className:'',iconSize:[36,36],iconAnchor:[18,18]}}),zIndexOffset:150}})
    .addTo(map)
    .bindPopup(`<b style="color:#f59e0b;font-size:14px;">STOPOVER</b><br>
      <b>{sname}</b><br>IATA:<b>{scode}</b> | {stier}<br>
      <span style="color:#94a3b8;">approx 1.5-2 hr layover</span>`);
  ['Leg 1','Leg 2'].forEach(function(lbl,i){{
    var midPt=i===0?leg1[Math.floor(leg1.length/2)]:leg2[Math.floor(leg2.length/2)];
    var legHtml = `<div class="flight-dist-pill" style="border-color:#fbbf24;box-shadow:0 4px 18px rgba(0,0,0,0.85),0 0 14px rgba(251,191,36,0.6);">` +
      `<span class="ico">&#x2708;</span>` +
      `<span class="dst" style="color:#fbbf24;">` + lbl + `</span>` +
    `</div>`;
    L.marker(midPt,{{icon:L.divIcon({{html:legHtml,className:'',iconSize:[0,0],iconAnchor:[0,0]}}),interactive:false,zIndexOffset:450}}).addTo(map);
  }});
"""
        info_stop = f"<div class='row' style='color:#f59e0b;'>&#x1F504; 1-Stop via {scode}</div>"
    else:
        stop_js = f"""
  // DIRECT FLIGHT: straight linear route from source to destination
  var arc = linearPath([{lat1},{lon1}],[{lat2},{lon2}],100);
  L.polyline(arc,{{color:'#38bdf8',weight:16,opacity:0.08}}).addTo(map);
  L.polyline(arc,{{color:'#38bdf8',weight:7, opacity:0.25}}).addTo(map);
  L.polyline(arc,{{color:'#fbbf24',weight:3, opacity:1.0,dashArray:'12 7'}}).addTo(map);
  var mid=arc[Math.floor(arc.length/2)];
  var distBadgeHtml = `<div class="flight-dist-pill">` +
    `<span class="ico">&#x2708;</span>` +
    `<span class="dst">{distance_km} km</span>` +
    `<span class="sep">&bull;</span>` +
    `<span class="dur">{duration_hrs} hrs</span>` +
  `</div>`;
  L.marker(mid,{{icon:L.divIcon({{html:distBadgeHtml,className:'',iconSize:[0,0],iconAnchor:[0,0]}}),interactive:false,zIndexOffset:650}}).addTo(map);
"""
        info_stop = "<div class='row' style='color:#34d399;'>&#x2705; Direct non-stop</div>"

    html_code = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0f172a;font-family:Segoe UI,sans-serif;}}
  #map{{width:100%;height:480px;border-radius:12px;border:2px solid #1e40af;box-shadow:0 8px 32px rgba(0,0,0,.7);}}
  .info-box{{position:absolute;bottom:18px;left:18px;z-index:1000;background:rgba(10,15,30,.95);
    border:1.5px solid #38bdf8;border-radius:12px;padding:12px 16px;font-size:13px;
    color:#f8fafc;min-width:210px;backdrop-filter:blur(10px);
    box-shadow:0 8px 24px rgba(0,0,0,.8),0 0 12px rgba(56,189,248,.3);pointer-events:none;}}
  .flight-dist-pill{{background:linear-gradient(135deg,rgba(15,23,42,.96) 0%,rgba(30,58,138,.94) 100%)!important;
    border:2.5px solid #38bdf8!important;border-radius:30px!important;padding:7px 18px!important;
    box-shadow:0 4px 22px rgba(0,0,0,.9),0 0 18px rgba(56,189,248,.75)!important;color:#fff!important;
    white-space:nowrap!important;pointer-events:none!important;display:inline-flex!important;
    align-items:center!important;gap:9px!important;transform:translate(-50%,-50%)!important;
    font-family:'Segoe UI',system-ui,sans-serif!important;}}
  .flight-dist-pill .ico{{font-size:17px;color:#fbbf24;filter:drop-shadow(0 0 5px #f59e0b);}}
  .flight-dist-pill .dst{{color:#ffffff;font-weight:900;font-size:15px;letter-spacing:0.4px;}}
  .flight-dist-pill .sep{{color:#64748b;font-weight:400;font-size:13px;}}
  .flight-dist-pill .dur{{color:#38bdf8;font-weight:900;font-size:15px;letter-spacing:0.4px;}}
  .info-box b{{color:#38bdf8;}} .row{{margin:3px 0;}}
  .leaflet-popup-content-wrapper{{background:#1e293b!important;color:#f8fafc!important;
    border-radius:10px!important;border:1px solid #3b82f6!important;font-size:13px!important;}}
  .leaflet-popup-content{{color:#f8fafc!important;}}
  .leaflet-popup-tip{{background:#1e293b!important;}}
  .leaflet-control-layers{{background:rgba(10,15,30,.92)!important;border:1px solid #334155!important;border-radius:8px!important;}}
  .leaflet-control-layers-base label{{color:#cbd5e1!important;margin:3px 0;}}
  .leaflet-control-zoom a{{background:rgba(10,15,30,.9)!important;color:#38bdf8!important;border:1px solid #334155!important;}}
  .leaflet-bar a:hover{{background:#1e3a5f!important;}}
  @keyframes pulseRing{{0%{{transform:scale(.6);opacity:.9;}}70%{{transform:scale(2);opacity:0;}}100%{{transform:scale(2);opacity:0;}}}}
  .pulse-ring{{border-radius:50%;animation:pulseRing 2s ease-out infinite;position:absolute;top:0;left:0;right:0;bottom:0;}}
</style>
</head>
<body>
<div id="map"></div>
<div class="info-box">
  <div class="row">&#x1F6EB; <b>{src_info["code"]}</b> &rarr; <b>{dst_info["code"]}</b></div>
  <div class="row">&#x1F4CF; <b>{distance_km} km</b> direct route</div>
  <div class="row">&#x23F1; Est. <b>{duration_hrs} hrs</b></div>
  {info_stop}
  <div class="row" style="color:#94a3b8;font-size:11px;">&#x2708; Live simulation active</div>
</div>
<script>
  var map=L.map("map",{{zoomControl:true,attributionControl:false}});
  var gSat=L.tileLayer("https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}",{{maxZoom:20}});
  var gStr=L.tileLayer("https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}",{{maxZoom:20}});
  var gTer=L.tileLayer("https://mt1.google.com/vt/lyrs=p&x={{x}}&y={{y}}&z={{z}}",{{maxZoom:20}});
  gSat.addTo(map);
  L.control.layers({{"&#x1F6F0; Satellite":gSat,"&#x1F5FA; Streets":gStr,"&#x26F0; Terrain":gTer}},null,{{position:"topright",collapsed:false}}).addTo(map);

  // fitBounds with maxZoom:8 prevents zooming in too close on short routes
  map.fitBounds([[{min_lat:.4f},{min_lon:.4f}],[{max_lat:.4f},{max_lon:.4f}]],{{padding:[55,55],maxZoom:8}});

  // Linear flight interpolation from p1 to p2 for realistic straight airways
  function linearPath(p1,p2,n){{
    var pts=[],dLat=p2[0]-p1[0],dLon=p2[1]-p1[1];
    for(var i=0;i<=n;i++){{
      var t=i/n;
      pts.push([p1[0]+dLat*t, p1[1]+dLon*t]);
    }}
    return pts;
  }}

  // Calculate compass heading (bearing in degrees) from p1 to p2
  function getHeading(p1,p2){{
    var dLat=p2[0]-p1[0];
    var dLon=(p2[1]-p1[1])*Math.cos(((p1[0]+p2[0])/2)*Math.PI/180);
    return (Math.atan2(dLon,dLat)*180/Math.PI+360)%360;
  }}

  {stop_js}

  // Origin (cyan pulse)
  var oH=`<div style="position:relative;width:36px;height:36px;">
    <div class="pulse-ring" style="background:#38bdf8;opacity:.55;"></div>
    <div class="pulse-ring" style="background:#38bdf8;opacity:.3;animation-delay:.7s;"></div>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
      width:18px;height:18px;background:#38bdf8;border-radius:50%;
      border:3px solid #fff;box-shadow:0 0 14px #38bdf8;"></div></div>`;
  L.marker([{lat1},{lon1}],{{icon:L.divIcon({{html:oH,className:"",iconSize:[36,36],iconAnchor:[18,18]}}),zIndexOffset:200}})
    .addTo(map).bindPopup(`<b style="color:#38bdf8;font-size:14px;">&#x1F6EB; DEPARTURE</b><br>
      <b>{src_info["name"]}</b><br>IATA:<b>{src_info["code"]}</b> Tier:{src_info["tier"]}<br>
      <span style="color:#94a3b8;">{lat1:.4f}N {lon1:.4f}E</span>`);

  // Destination (red pulse)
  var dH=`<div style="position:relative;width:36px;height:36px;">
    <div class="pulse-ring" style="background:#f43f5e;opacity:.55;animation-delay:.4s;"></div>
    <div class="pulse-ring" style="background:#f43f5e;opacity:.3;animation-delay:1.1s;"></div>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
      width:18px;height:18px;background:#f43f5e;border-radius:50%;
      border:3px solid #fff;box-shadow:0 0 14px #f43f5e;"></div></div>`;
  L.marker([{lat2},{lon2}],{{icon:L.divIcon({{html:dH,className:"",iconSize:[36,36],iconAnchor:[18,18]}}),zIndexOffset:200}})
    .addTo(map).bindPopup(`<b style="color:#f43f5e;font-size:14px;">&#x1F6EC; ARRIVAL</b><br>
      <b>{dst_info["name"]}</b><br>IATA:<b>{dst_info["code"]}</b> Tier:{dst_info["tier"]}<br>
      <span style="color:#94a3b8;">{lat2:.4f}N {lon2:.4f}E</span>`);

  // Animated airplane following linear flight path with dynamic heading orientation
  var initHeading=arc.length>1?getHeading(arc[0],arc[1]):0;
  var planeHtml=`<div id="sim-plane" style="width:32px;height:32px;transform:rotate(`+initHeading.toFixed(1)+`deg);transform-origin:center center;transition:transform .12s linear;"><svg width="32" height="32" viewBox="0 0 24 24" fill="#fbbf24" style="filter:drop-shadow(0 0 6px #fbbf24) drop-shadow(0 0 12px #f59e0b);display:block;"><path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg></div>`;
  var planeMkr=L.marker(arc[0],{{icon:L.divIcon({{
    html:planeHtml,
    className:"",iconSize:[32,32],iconAnchor:[16,16]}}),zIndexOffset:1000}}).addTo(map);

  var step=0;
  (function fly(){{
    step=(step+1)%arc.length;
    var curr=arc[step];
    var next=arc[(step+1)%arc.length];
    planeMkr.setLatLng(curr);
    var el=document.getElementById("sim-plane");
    if(el && next){{
      var h=getHeading(curr,next);
      el.style.transform="rotate("+h.toFixed(1)+"deg)";
    }}
    setTimeout(fly,(step<5||step>arc.length-5)?150:65);
  }})();
</script>
</body>
</html>"""
    return html_code



# ─────────────────────────────────────────────────────────────
#  MAIN VIEW LOGIC
# ─────────────────────────────────────────────────────────────
if not st.session_state.has_calculated:
    # WELCOME BANNER
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#0f2a47 100%);
                border:1.5px solid #1e40af; border-radius:16px; padding:36px 28px;
                text-align:center; margin:20px 0;">
        <div style="font-size:56px; margin-bottom:10px;">✈️</div>
        <h2 style="color:#38bdf8 !important; font-size:1.8rem; margin-bottom:10px;">
            Ready to Analyze Your Airfare Route
        </h2>
        <p style="color:#94a3b8; font-size:15px; max-width:640px; margin:0 auto 22px auto;">
            Currently configured:
            <b style="color:#f8fafc;">{origin_city} ({AIRPORTS[origin_city]['code']})</b> ➔
            <b style="color:#f8fafc;">{dest_city} ({AIRPORTS[dest_city]['code']})</b> on
            <b style="color:#f8fafc;">{selected_airline}</b>
            ({flight_class} Class, {days_left} days before departure).
        </p>
        <div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap; margin-bottom:24px;">
            <div style="background:#1e3a5f; border-radius:10px; padding:12px 20px; min-width:130px;">
                <div style="color:#38bdf8; font-size:20px; font-weight:800;">{route_info['distance_km']} km</div>
                <div style="color:#64748b; font-size:11px; margin-top:2px;">GREAT-CIRCLE DIST.</div>
            </div>
            <div style="background:#1e3a5f; border-radius:10px; padding:12px 20px; min-width:130px;">
                <div style="color:#fbbf24; font-size:20px; font-weight:800;">{route_info['est_duration']} hrs</div>
                <div style="color:#64748b; font-size:11px; margin-top:2px;">EST. FLIGHT TIME</div>
            </div>
            <div style="background:#1e3a5f; border-radius:10px; padding:12px 20px; min-width:130px;">
                <div style="color:#34d399; font-size:20px; font-weight:800;">{route_info['co2_kg']} kg</div>
                <div style="color:#64748b; font-size:11px; margin-top:2px;">CO₂ FOOTPRINT</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
    with c_btn2:
        if st.button("🚀 Calculate & Show Airfare Breakdown", type="primary", use_container_width=True):
            st.session_state.has_calculated = True
            st.rerun()

else:
    # RECALCULATE ROW
    rcol1, rcol2 = st.columns([4, 1.2])
    with rcol1:
        st.markdown(
            f"<span style='color:#64748b; font-size:13px;'>Showing results for: "
            f"<b style='color:#f8fafc;'>{origin_city} ➔ {dest_city}</b> | "
            f"<b style='color:#38bdf8;'>{selected_airline}</b> | "
            f"<b style='color:#fbbf24;'>{flight_class}</b> | "
            f"{days_left} days left</span>",
            unsafe_allow_html=True
        )
    with rcol2:
        if st.button("🔄 Recalculate / Update", use_container_width=True):
            st.session_state.has_calculated = True
            st.rerun()

    # ── TABS ──
    tab_fare, tab_multi, tab_map, tab_benchmark, tab_insights = st.tabs([
        "🎯 Fare Prediction & Bill",
        "🏢 Multi-Airline Price Matrix",
        "🗺️ Interactive Flight Map",
        "📊 AI Model Benchmark & Performance",
        "🔍 Feature Drivers & Intelligence"
    ])

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — FARE PREDICTION & COMPONENT BILL
    # ══════════════════════════════════════════════════════════
    with tab_fare:
        if pipeline is None:
            st.error("ML model not loaded. Please ensure models/best_model.joblib exists.")
        else:
            if is_compare_all:
                df_all_pred = get_multi_carrier_comparison(
                    pipeline, origin_city, dest_city, flight_class, dep_time, arr_time, days_left
                )
                best_carrier_row = df_all_pred.iloc[0]
                eval_airline = best_carrier_row['Airline']
                flight_result = predict_single_flight(
                    pipeline, origin_city, dest_city, eval_airline,
                    flight_class, dep_time, arr_time, days_left
                )
                fare_val = flight_result['price']
                min_fare = df_all_pred['Estimated_Fare'].min()
                max_fare = df_all_pred['Estimated_Fare'].max()
                avg_fare = df_all_pred['Estimated_Fare'].mean()
            else:
                eval_airline = selected_airline
                flight_result = predict_single_flight(
                    pipeline, origin_city, dest_city, selected_airline,
                    flight_class, dep_time, arr_time, days_left
                )
                fare_val = flight_result['price']
                min_fare, max_fare, avg_fare = fare_val, fare_val, fare_val

            breakdown = flight_result['breakdown']
            advisory = get_booking_advisory(days_left, fare_val, flight_class)

            # ── Query Live Real-Time Market Data ──
            live_data = fetch_realtime_flights(
                AIRPORTS[origin_city]['code'], AIRPORTS[dest_city]['code'],
                days_left=days_left, cabin_class=flight_class
            )
            carrier_live_match = get_selected_carrier_live_info(live_data, selected_airline)
            live_price = carrier_live_match.get('live_price')
            variance_info = compute_market_variance(fare_val, live_price)

            # ── Top metric banner (4 Columns with Real-Time Comparison) ──
            m_col1, m_col2, m_col3, m_col4 = st.columns([1.3, 1.3, 1.3, 1.2])

            with m_col1:
                with st.container(border=True):
                    if is_compare_all:
                        st.metric(
                            label="🤖 ML Predicted Range (All Carriers)",
                            value=f"₹ {min_fare:,.0f} – ₹ {max_fare:,.0f}",
                            delta=f"Best: {eval_airline} (Avg ₹{avg_fare:,.0f})"
                        )
                        st.caption(
                            f"6-Carrier Fare Spread • Baseline: **₹ {fare_val:,.0f}**"
                        )
                    else:
                        st.metric(
                            label=f"🤖 ML Predicted Airfare",
                            value=f"₹ {fare_val:,.0f}",
                            delta=f"{selected_airline} • {flight_class}"
                        )
                        st.caption(
                            f"Confidence Range (±MAE): **₹ {max(1500, fare_val - 1819):,.0f} – ₹ {fare_val + 1819:,.0f}**"
                        )

            with m_col2:
                with st.container(border=True):
                    if is_compare_all and live_price:
                        delta_str = f"{variance_info['variance_pct']:+.1f}% vs Best ML"
                        lowest_c = live_data.get('flights', [{}])[0].get('airline', 'OTA')
                        st.metric(
                            label=f"🔴 Best Live Market ({lowest_c})",
                            value=f"₹ {live_price:,.0f}",
                            delta=delta_str,
                            delta_color="inverse" if variance_info['variance_pct'] > 0 else "normal"
                        )
                        st.caption(f"🟢 Lowest online fare on {live_data['date_str']} ({live_data.get('flight_count', 0)} flights)")
                    elif carrier_live_match.get("has_selected"):
                        delta_str = f"{variance_info['variance_pct']:+.1f}% vs ML"
                        st.metric(
                            label=f"🔴 Live {selected_airline} (Google Flights)",
                            value=f"₹ {live_price:,.0f}",
                            delta=delta_str,
                            delta_color="inverse" if variance_info['variance_pct'] > 0 else "normal"
                        )
                        st.caption(f"🟢 Live {selected_airline} seat on {live_data['date_str']}")
                    elif live_price:
                        delta_str = f"{variance_info['variance_pct']:+.1f}% vs ML"
                        lowest_c = carrier_live_match.get('lowest_carrier', 'OTA')
                        st.metric(
                            label=f"🔴 Route Lowest Live ({lowest_c})",
                            value=f"₹ {live_price:,.0f}",
                            delta=delta_str,
                            delta_color="inverse" if variance_info['variance_pct'] > 0 else "normal"
                        )
                        st.caption(f"ℹ️ {selected_airline} has no published seat on this date")
                    else:
                        st.metric(
                            label="🔴 Live Market (Google Flights)",
                            value="₹ Benchmark",
                            delta="Live feed idle"
                        )
                        st.caption("No direct OTA seat on exact date")

            with m_col3:
                with st.container(border=True):
                    st.markdown(
                        f"<span style='color:{advisory['color']}; font-weight:800; font-size:15px;'>"
                        f"{advisory['icon']} {advisory['status']}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-size:13px; color:#94a3b8; margin-top:6px;'>"
                        f"{advisory['action']}</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"**Expected Trend:** {advisory['expected_trend']}")

            with m_col4:
                with st.container(border=True):
                    st.markdown(
                        "<span style='color:#64748b; font-weight:700; font-size:11px;'>"
                        "ROUTE & ACCURACY INDEX</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"**{AIRPORTS[origin_city]['code']} ➔ {AIRPORTS[dest_city]['code']}**"
                        f" • `{route_info['distance_km']} km`"
                    )
                    st.markdown(f"**Status:** `{variance_info['status']}`")
                    st.caption(f"Duration: **{route_info['est_duration']} hrs** • **{route_info['rec_stops'].capitalize()}** stop")

            st.markdown("---")

            # ── Bill Breakdown + Donut ──
            b_col1, b_col2 = st.columns([1.3, 1.2])

            with b_col1:
                with st.container(border=True):
                    st.subheader("🧾 Component Cost Breakdown (Bill of Airfare)")
                    st.caption(
                        "Detailed decomposition of airfare into base operating tariff, "
                        "fuel surcharges, and statutory aviation fees."
                    )

                    breakdown_items = [
                        ("Base Airfare", "Operations & Fleet Margin",
                         breakdown["Base Fare"], "#38bdf8", "🔵"),
                        ("Fuel Surcharge (YQ)", "Dynamic Demand Surcharge",
                         breakdown["Fuel & Dynamic Surcharge (YQ)"], "#fbbf24", "🟡"),
                        ("Airport Fees (UDF/PSF)", "Airport Development Fee + ASF",
                         breakdown["Airport Development Fee (UDF/PSF)"], "#34d399", "🟢"),
                        ("Statutory GST", "Goods & Services Tax on Aviation",
                         breakdown["Goods & Services Tax (GST)"], "#c084fc", "🟣"),
                    ]

                    # ── Custom HTML linear progress bars (color-matched, animated) ──
                    bars_html = "<div style='margin-top:10px;'>"
                    for short_label, desc, amt, col_code, emoji in breakdown_items:
                        b_pct = (amt / fare_val) * 100
                        bars_html += f"""
                        <div style="margin-bottom:18px;">
                            <div style="display:flex; justify-content:space-between;
                                        align-items:baseline; margin-bottom:6px;">
                                <div>
                                    <span style="color:{col_code}; font-size:15px;">{emoji}</span>
                                    <span style="color:#f8fafc; font-weight:700;
                                                 font-size:13px; margin-left:4px;">
                                        {short_label}
                                    </span>
                                    <span style="color:#64748b; font-size:11px;
                                                 margin-left:6px;">— {desc}</span>
                                </div>
                                <div style="text-align:right; min-width:110px;">
                                    <span style="color:{col_code}; font-weight:800;
                                                 font-size:14px;">₹ {amt:,.0f}</span>
                                    <span style="color:#64748b; font-size:11px;
                                                 margin-left:4px;">({b_pct:.1f}%)</span>
                                </div>
                            </div>
                            <div style="background:#0f172a; border-radius:6px;
                                        height:10px; width:100%; overflow:hidden;
                                        border:1px solid #1e293b;">
                                <div style="
                                    height:100%;
                                    width:{b_pct:.2f}%;
                                    background:linear-gradient(90deg, {col_code}cc, {col_code});
                                    border-radius:6px;
                                    box-shadow: 0 0 8px {col_code}88;
                                    animation: growBar 1.2s ease-out forwards;
                                    transform-origin: left;
                                "></div>
                            </div>
                        </div>"""
                    bars_html += "</div>"

                    # Stacked bar showing all components together
                    stack_parts = ""
                    for short_label, desc, amt, col_code, emoji in breakdown_items:
                        b_pct = (amt / fare_val) * 100
                        stack_parts += f"""
                        <div title="{short_label}: ₹{amt:,.0f} ({b_pct:.1f}%)"
                             style="width:{b_pct:.2f}%; height:100%;
                                    background:{col_code}; transition:width 0.5s;"></div>"""

                    bars_html += f"""
                    <div style="margin:14px 0 8px 0;">
                        <div style="color:#64748b; font-size:11px; font-weight:600;
                                    margin-bottom:5px; letter-spacing:0.5px;">
                            COMBINED FARE COMPOSITION
                        </div>
                        <div style="display:flex; height:16px; border-radius:8px;
                                    overflow:hidden; border:1px solid #1e293b;
                                    box-shadow:0 2px 8px rgba(0,0,0,0.4);">
                            {stack_parts}
                        </div>
                        <div style="display:flex; gap:16px; margin-top:7px; flex-wrap:wrap;">"""

                    for short_label, _, _, col_code, emoji in breakdown_items:
                        bars_html += f"""
                            <span style="font-size:11px; color:#94a3b8;">
                                <span style="color:{col_code};">■</span> {short_label}
                            </span>"""
                    bars_html += "</div></div>"

                    st.markdown(f"""
                    <style>
                    @keyframes growBar {{
                        from {{ width: 0%; }}
                    }}
                    </style>
                    {bars_html}
                    """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown(
                        f"### Total Invoice Amount: "
                        f"<span style='color:#38bdf8;'>₹ {fare_val:,.2f}</span>",
                        unsafe_allow_html=True
                    )

            with b_col2:
                with st.container(border=True):
                    st.subheader("📊 Component Cost Share")
                    donut_labels = [
                        "Base Fare", "Fuel Surcharge (YQ)",
                        "Airport Fees (UDF/PSF)", "Statutory GST"
                    ]
                    donut_values = [
                        breakdown["Base Fare"],
                        breakdown["Fuel & Dynamic Surcharge (YQ)"],
                        breakdown["Airport Development Fee (UDF/PSF)"],
                        breakdown["Goods & Services Tax (GST)"]
                    ]

                    fig_donut = go.Figure(data=[go.Pie(
                        labels=donut_labels, values=donut_values, hole=0.52,
                        marker=dict(colors=['#38bdf8', '#fbbf24', '#34d399', '#c084fc']),
                        textinfo="label+percent",
                        textfont=dict(color="#ffffff", size=12),
                        hovertemplate="<b>%{label}</b><br>₹ %{value:,.0f}<br>%{percent}<extra></extra>"
                    )])
                    fig_donut.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=10, b=10, l=10, r=10), height=295,
                        showlegend=False,
                        annotations=[dict(
                            text=f"<b>₹{fare_val:,.0f}</b>",
                            x=0.5, y=0.5, font_size=14, showarrow=False,
                            font=dict(color="#f8fafc")
                        )]
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

            st.markdown("---")

            # ── Booking Window Sensitivity ──
            st.subheader("📈 Dynamic Booking Window Sensitivity Simulation")
            st.write("Yield management price acceleration — 50 days → 1 day before departure:")

            days_seq = np.arange(1, 51)
            sim_prices = []
            for d in days_seq:
                sim_res = predict_single_flight(
                    pipeline, origin_city, dest_city, eval_airline if is_compare_all else selected_airline,
                    flight_class, dep_time, arr_time, int(d)
                )
                sim_prices.append(sim_res['price'])

            df_sim = pd.DataFrame({
                "Days Before Departure": days_seq,
                "Estimated Price (INR)": sim_prices
            })

            fig_curve = px.line(
                df_sim, x="Days Before Departure", y="Estimated Price (INR)",
                markers=True,
                title=f"Airfare Yield Curve: {origin_city} ➔ {dest_city} ({selected_airline} - {flight_class})",
                color_discrete_sequence=["#38bdf8"]
            )
            fig_curve.add_vline(
                x=days_left, line_dash="dash", line_color="#f43f5e",
                annotation_text=f"Selected: Day {days_left}",
                annotation_font_color="#f43f5e"
            )
            # Shade high-price zone (last 10 days)
            fig_curve.add_vrect(
                x0=1, x1=10,
                fillcolor="rgba(244,63,94,0.08)", line_width=0,
                annotation_text="⚠️ Surge Zone", annotation_position="top left",
                annotation_font_color="#f43f5e"
            )
            fig_curve.update_xaxes(autorange="reversed", title="Days Before Departure")
            fig_curve.update_yaxes(title="Estimated Airfare (₹)")
            fig_curve.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=370,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b")
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            # ── PDF Download ──
            st.markdown("---")
            pdf_carrier = f'{eval_airline} (Lowest Fare)' if is_compare_all else selected_airline
            pdf_bytes = generate_flight_quotation_pdf(
                origin_city=origin_city, dest_city=dest_city,
                src_info=AIRPORTS[origin_city], dst_info=AIRPORTS[dest_city],
                selected_airline=pdf_carrier, flight_class=flight_class,
                dep_time=dep_time, arr_time=arr_time, days_left=days_left,
                route_info=route_info, fare_val=fare_val,
                breakdown=breakdown, advisory=advisory
            )

            st.download_button(
                label="📄 Download Official Airfare Quotation (Executive PDF Report)",
                data=pdf_bytes,
                file_name=f"Flight_Quotation_{AIRPORTS[origin_city]['code']}_{AIRPORTS[dest_city]['code']}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            st.caption(
                "💡 Official 1-Page PDF Quotation — compatible with all phones, tablets & computers. "
                "Contains verified flight specs, itemised cost breakdown, and AI valuation advisory."
            )

    # ══════════════════════════════════════════════════════════
    #  TAB 2 — MULTI-AIRLINE PRICE MATRIX (pure HTML cards)
    # ══════════════════════════════════════════════════════════
    with tab_multi:
        st.subheader(f"🏢 Multi-Carrier Pricing Matrix: {origin_city} ➔ {dest_city}")
        st.write(
            f"Compare real-time online flight tariffs directly against trained Supervised ML model baselines for "
            f"**{flight_class} Class** with **{days_left} days left**:"
        )

        src_iata = AIRPORTS[origin_city]['code']
        dst_iata = AIRPORTS[dest_city]['code']
        dep_date_obj = datetime.now() + timedelta(days=days_left)
        dep_date_str = dep_date_obj.strftime("%Y-%m-%d")
        live_gflights_url = f"https://www.google.com/travel/flights?q=Flights%20to%20{dst_iata}%20from%20{src_iata}%20on%20{dep_date_str}%20oneway"

        # ── Toggle View Modes ──
        mode_c1, mode_c2 = st.columns([3.2, 1.8])
        with mode_c1:
            pricing_source = st.radio(
                "Select Pricing Intelligence Source:",
                ["🔴 Live Real-Time Tariffs (Google Flights Feed)", "🤖 Supervised ML Model Benchmark"],
                horizontal=True
            )
        with mode_c2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            st.link_button("🌐 Open Live Search on Google Flights", live_gflights_url, use_container_width=True)

        # Baseline ML multi-carrier comparison
        df_multi = get_multi_carrier_comparison(
            pipeline, origin_city, dest_city, flight_class, dep_time, arr_time, days_left
        )

        carrier_colors = {
            "IndiGo":        "#38bdf8",
            "Air India":     "#f43f5e",
            "Vistara":       "#a855f7",
            "AirAsia India": "#fb923c",
            "SpiceJet":      "#fbbf24",
            "GO FIRST":      "#34d399",
            "Akasa Air":     "#c084fc",
            "Air India Express": "#f97316"
        }

        if "Live Real-Time" in pricing_source:
            with st.spinner(f"Querying live online flights for {src_iata} ➔ {dst_iata} on {dep_date_str}..."):
                live_res = fetch_realtime_flights(
                    src_iata, dst_iata, days_left=days_left, cabin_class=flight_class
                )

            if live_res.get("success") and live_res.get("flights"):
                # Top metrics
                lm_col1, lm_col2, lm_col3 = st.columns(3)
                with lm_col1:
                    st.metric("Lowest Live Market Fare", f"₹ {live_res['lowest_price']:,}", "Live OTA Seat")
                with lm_col2:
                    st.metric("Live Flights Online", f"{live_res['flight_count']} options", f"Travel Date: {live_res['date_str']}")
                with lm_col3:
                    st.metric("Live Feed Source", "🟢 Google Flights API", "15m Smart Cache")

                if selected_airline in live_res["carrier_best"]:
                    st.success(f"✅ Found live flights for your preferred carrier: **{selected_airline}** at **₹ {live_res['carrier_best'][selected_airline]['price']:,}**")
                else:
                    st.info(f"ℹ️ **{selected_airline}** does not have published direct seats on Google Flights for **{dep_date_str}**. Showing available active carriers on this route below.")

                st.markdown("#### ✈️ Live Bookable Airline Tariffs (Cheapest Found per Carrier)")

                # Build live carrier cards
                live_cards_html = []
                cheapest_live_price = live_res["lowest_price"]
                for c_name, c_info in live_res["carrier_best"].items():
                    c_col = carrier_colors.get(c_name, "#60a5fa")
                    is_cheapest = (c_info["price"] == cheapest_live_price)
                    is_user_selected = (c_name == selected_airline)
                    if is_user_selected and is_cheapest:
                        border_c = "#38bdf8"
                        shadow_c = "0 0 20px rgba(56,189,248,0.45)"
                        badge_str = "<div style='background:#0284c7;color:#fff;padding:3px 10px;border-radius:10px;font-size:10px;font-weight:800;letter-spacing:0.5px;white-space:nowrap;'>⭐ YOUR AIRLINE (BEST)</div>"
                    elif is_user_selected:
                        border_c = "#38bdf8"
                        shadow_c = "0 0 20px rgba(56,189,248,0.45)"
                        badge_str = "<div style='background:#0284c7;color:#fff;padding:3px 10px;border-radius:10px;font-size:10px;font-weight:800;letter-spacing:0.5px;white-space:nowrap;'>⭐ YOUR AIRLINE</div>"
                    elif is_cheapest:
                        border_c = "#10b981"
                        shadow_c = "0 0 18px rgba(16,185,129,0.3)"
                        badge_str = "<div style='background:#10b981;color:#fff;padding:3px 10px;border-radius:10px;font-size:10px;font-weight:800;letter-spacing:0.5px;white-space:nowrap;'>🏆 CHEAPEST LIVE</div>"
                    else:
                        border_c = "#334155"
                        shadow_c = "0 2px 10px rgba(0,0,0,0.4)"
                        badge_str = "<div></div>"

                    card = f"""
                    <div style="background:#1e293b;border:1.5px solid {border_c};border-radius:12px;padding:16px;min-height:160px;box-shadow:{shadow_c};box-sizing:border-box;">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                            <span style="color:{c_col};font-size:17px;font-weight:800;">{c_name}</span>
                            {badge_str}
                        </div>
                        <div style="color:#64748b;font-size:11px;margin-bottom:8px;">{c_info['aircraft']} &bull; {c_info['stops_label']}</div>
                        <div style="color:#38bdf8;font-size:26px;font-weight:800;margin-bottom:8px;">&#8377; {c_info['price']:,}</div>
                        <div style="color:#94a3b8;font-size:12px;">&#9200; <b style="color:#e2e8f0;">{c_info['duration_hours']} hrs</b> &nbsp;|&nbsp; {c_info['departure_time']} &rarr; {c_info['arrival_time']}</div>
                    </div>"""
                    live_cards_html.append(card)

                while len(live_cards_html) % 3 != 0:
                    live_cards_html.append("<div></div>")

                grid_live_doc = f"""<!DOCTYPE html>
                <html><head><meta charset="utf-8"/><style>
                *{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Segoe UI',sans-serif;}}
                .g{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}}
                </style></head><body><div class="g">{''.join(live_cards_html)}</div></body></html>"""
                components.html(grid_live_doc, height=200 if len(live_cards_html) <= 3 else 390, scrolling=False)

                st.markdown("---")
                st.markdown("#### 📊 Live Market Fare vs ML Prediction by Carrier")

                comp_carrier_data = []
                for _, ml_r in df_multi.iterrows():
                    c_name = ml_r['Airline']
                    ml_f = ml_r['Estimated_Fare']
                    live_c_info = live_res["carrier_best"].get(c_name)
                    live_f = live_c_info["price"] if live_c_info else None
                    comp_carrier_data.append({
                        "Airline": c_name,
                        "ML Predicted Fare (₹)": ml_f,
                        "Live Google Flights Fare (₹)": live_f
                    })

                # Also add any airlines in live that were not in ML dataset (e.g. Akasa Air, Air India Express)
                for c_name, c_info in live_res["carrier_best"].items():
                    if not any(row["Airline"] == c_name for row in comp_carrier_data):
                        comp_carrier_data.append({
                            "Airline": c_name,
                            "ML Predicted Fare (₹)": None,
                            "Live Google Flights Fare (₹)": c_info["price"]
                        })

                df_carrier_comp = pd.DataFrame(comp_carrier_data)

                fig_live_vs_ml = go.Figure()
                fig_live_vs_ml.add_trace(go.Bar(
                    x=df_carrier_comp["Airline"], y=df_carrier_comp["ML Predicted Fare (₹)"],
                    name="🤖 ML Prediction Baseline", marker_color="#3b82f6",
                    text=df_carrier_comp["ML Predicted Fare (₹)"].apply(lambda x: f"₹{x:,.0f}" if pd.notnull(x) else "N/A"),
                    textposition="auto"
                ))
                fig_live_vs_ml.add_trace(go.Bar(
                    x=df_carrier_comp["Airline"], y=df_carrier_comp["Live Google Flights Fare (₹)"],
                    name="🔴 Live Market (Google Flights)", marker_color="#10b981",
                    text=df_carrier_comp["Live Google Flights Fare (₹)"].apply(lambda x: f"₹{x:,.0f}" if pd.notnull(x) else "No Live Seat"),
                    textposition="auto"
                ))
                fig_live_vs_ml.update_layout(
                    barmode="group",
                    title=f"Live Online Airfares vs Supervised ML Estimation ({origin_city} ➔ {dest_city})",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=350, margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(title="Fare (₹)", gridcolor="#1e293b"),
                    xaxis=dict(gridcolor="#1e293b"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_live_vs_ml, use_container_width=True)

                st.markdown("#### 📋 Real-Time Flights Schedule (Online OTA Feed)")
                df_flights_table = pd.DataFrame(live_res["flights"])
                st.dataframe(
                    df_flights_table[['airline', 'price', 'stops_label', 'duration_hours', 'departure_time', 'arrival_time', 'aircraft', 'source']]
                    .rename(columns={
                        'airline': 'Airline', 'price': 'Live_Price', 'stops_label': 'Stops',
                        'duration_hours': 'Duration_Hrs', 'departure_time': 'Departure',
                        'arrival_time': 'Arrival', 'aircraft': 'Aircraft', 'source': 'Source'
                    }).style.format({'Live_Price': '₹{:,.0f}', 'Duration_Hrs': '{:.1f} hrs'})
                    .highlight_min(subset=['Live_Price'], color='#064e3b'),
                    use_container_width=True
                )
            else:
                st.info(f"ℹ️ Google Flights returned: {live_res.get('error', 'No live flights found for this specific route.')} You can check the live search directly via the button above.")

        else:
            # ── Standard ML Matrix ──
            sort_choice = st.radio(
                "Sort Carriers By:",
                ["Lowest Price First", "Shortest Duration", "Carrier Name"],
                horizontal=True
            )

            if sort_choice == "Lowest Price First":
                df_multi = df_multi.sort_values("Estimated_Fare").reset_index(drop=True)
            elif sort_choice == "Shortest Duration":
                df_multi = df_multi.sort_values("Duration_Hours").reset_index(drop=True)
            else:
                df_multi = df_multi.sort_values("Airline").reset_index(drop=True)

            lowest_fare = df_multi['Estimated_Fare'].min()

            cards_html_parts = []
            for _, row in df_multi.iterrows():
                ac = carrier_colors.get(row['Airline'], "#60a5fa")
                is_best = (row['Estimated_Fare'] == lowest_fare)
                border_col = "#10b981" if is_best else "#334155"
                shadow = "0 0 20px rgba(16,185,129,0.30)" if is_best else "0 2px 10px rgba(0,0,0,0.4)"
                badge = (
                    "<div style='background:#10b981;color:#fff;padding:3px 10px;border-radius:10px;font-size:10px;font-weight:800;letter-spacing:0.5px;white-space:nowrap;'>🏆 BEST FARE</div>"
                    if is_best else "<div></div>"
                )
                card = f"""
                <div style="background:#1e293b;border:1.5px solid {border_col};border-radius:12px;padding:18px 16px;min-height:160px;box-shadow:{shadow};box-sizing:border-box;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;gap:6px;">
                        <div>
                            <span style="color:{ac};font-size:17px;font-weight:800;line-height:1.2;">{row['Airline']}</span>
                            <span style="color:#64748b;font-size:12px;"> ({row['Code']})</span>
                        </div>
                        {badge}
                    </div>
                    <div style="color:#64748b;font-size:11px;margin-bottom:10px;">{row['Type']} &bull; {row['Badge']}</div>
                    <div style="color:#38bdf8;font-size:26px;font-weight:800;margin-bottom:8px;">&#8377; {int(row['Estimated_Fare']):,}</div>
                    <div style="color:#94a3b8;font-size:12px;">&#9200; <b style="color:#e2e8f0;">{row['Duration_Hours']} hrs</b> &nbsp;|&nbsp; Stops: <b style="color:#e2e8f0;">{row['Stops']}</b></div>
                </div>"""
                cards_html_parts.append(card)

            while len(cards_html_parts) % 3 != 0:
                cards_html_parts.append("<div></div>")

            grid_full_html = f"""<!DOCTYPE html>
            <html><head><meta charset="utf-8"/><style>
            *{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'Segoe UI',Arial,sans-serif;padding:4px 0;}}
            .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}}
            </style></head><body><div class="grid">{''.join(cards_html_parts)}</div></body></html>"""
            components.html(grid_full_html, height=420, scrolling=False)

            st.markdown("---")
            st.markdown("#### 📋 Comprehensive Carrier Comparison Table")

            fig_compare = px.bar(
                df_multi, x="Airline", y="Estimated_Fare",
                color="Airline",
                color_discrete_map=carrier_colors,
                title=f"Fare Comparison: {origin_city} ➔ {dest_city} ({flight_class})",
                text=df_multi['Estimated_Fare'].apply(lambda x: f"₹{x:,.0f}"),
                template="plotly_dark"
            )
            fig_compare.update_traces(textposition="outside", textfont=dict(color="#f8fafc", size=11))
            fig_compare.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=320, showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b", title="Estimated Fare (₹)")
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            st.dataframe(
                df_multi.style.format({
                    'Estimated_Fare': '₹{:,.0f}',
                    'Duration_Hours': '{:.1f} hrs'
                }).highlight_min(subset=['Estimated_Fare'], color='#064e3b')
                 .highlight_max(subset=['Estimated_Fare'], color='#4c0519'),
                use_container_width=True
            )

    # ══════════════════════════════════════════════════════════
    #  TAB 3 — INTERACTIVE FLIGHT MAP
    # ══════════════════════════════════════════════════════════
    with tab_map:
        st.subheader(f"🗺️ Geospatial Flight Corridor: {origin_city} ➔ {dest_city}")
        st.caption(
            f"Great-circle geodesic arc connecting "
            f"**{AIRPORTS[origin_city]['name']}** [{AIRPORTS[origin_city]['code']}] "
            f"to **{AIRPORTS[dest_city]['name']}** [{AIRPORTS[dest_city]['code']}]"
        )

        src_lat, src_lon = AIRPORTS[origin_city]['lat'], AIRPORTS[origin_city]['lon']
        dst_lat, dst_lon = AIRPORTS[dest_city]['lat'], AIRPORTS[dest_city]['lon']
        gmaps_url = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={src_lat},{src_lon}&destination={dst_lat},{dst_lon}"
        )

        gm_col1, gm_col2 = st.columns([3.5, 1.5])
        with gm_col1:
            st.markdown(
                "💡 Use the **layer switcher** (top-right) to toggle between "
                "Google Satellite, Streets, Terrain, and Traffic views."
            )
        with gm_col2:
            st.link_button("🌐 Open in Google Maps", gmaps_url, use_container_width=True)

        # Resolve stopover hub airport if flight has 1 stop
        stopover_info = None
        if route_info.get('rec_stops') == 'one':
            for hub_key in [route_info.get('proxy_src'), route_info.get('proxy_dst')]:
                if hub_key and hub_key in AIRPORTS and hub_key not in (origin_city, dest_city):
                    stopover_info = AIRPORTS[hub_key]
                    break

        map_html = generate_google_maps_html(
            AIRPORTS[origin_city], AIRPORTS[dest_city],
            route_info['distance_km'], route_info['est_duration'],
            stopover_info=stopover_info
        )
        components.html(map_html, height=500, scrolling=False)

        st.markdown("---")
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        with g_col1:
            st.metric("Great-Circle Distance", f"{route_info['distance_km']} km",
                      f"≈ {route_info['distance_miles']} miles")
        with g_col2:
            st.metric("Compass Bearing", f"{route_info['bearing']}°", "True North")
        with g_col3:
            alt = "FL340 (34,000 ft)" if route_info['distance_km'] > 600 else "FL240 (24,000 ft)"
            st.metric("Cruising Altitude", alt, "~Mach 0.78")
        with g_col4:
            st.metric("CO₂ Carbon Footprint", f"{route_info['co2_kg']} kg", "Per Passenger")

    # ══════════════════════════════════════════════════════════
    #  TAB 4 — MODEL COMPARISON (SIH BENCHMARK)
    # ══════════════════════════════════════════════════════════
    with tab_benchmark:
        st.subheader("📊 Machine Learning Performance & Model Benchmark")
        st.write("Comprehensive cross-algorithm performance benchmark evaluated on **300,153 validated flights** (80/20 train-test partition):")

        if metrics_dict:
            metrics_df = (
                pd.DataFrame(metrics_dict).T
                .reset_index()
                .rename(columns={'index': 'Model'})
            )

            st.dataframe(
                metrics_df.style.format({
                    'MAE': '₹{:,.2f}', 'RMSE': '₹{:,.2f}',
                    'MAPE': '{:.2f}%', 'R2': '{:.4f}',
                    'Training_Time_s': '{:.2f}s'
                }).highlight_max(subset=['R2'], color='#064e3b')
                 .highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#064e3b'),
                use_container_width=True
            )

            st.markdown("### 📈 Interactive Regression Diagnostics")

            # R² bar chart
            fig_bench = go.Figure()
            colors_bench = ['#475569', '#475569', '#0ea5e9', '#34d399', '#64748b']
            fig_bench.add_trace(go.Bar(
                x=metrics_df['Model'], y=metrics_df['R2'],
                name='R² Score',
                marker_color=colors_bench[:len(metrics_df)],
                text=metrics_df['R2'].round(4), textposition='outside',
                textfont=dict(color="#f8fafc")
            ))
            fig_bench.update_layout(
                title="Algorithm Comparison: R² Score (Higher is Better)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=320,
                yaxis=dict(range=[0.85, 1.0], gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_bench, use_container_width=True)

            st.markdown("#### 📸 Publication Diagnostic Plots")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                pva_img = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
                if os.path.exists(pva_img):
                    st.image(pva_img, caption="Predicted vs Actual Fare Scatter Plot",
                             use_container_width=True)
            with pcol2:
                res_img = os.path.join(PLOTS_DIR, "residuals_distribution.png")
                if os.path.exists(res_img):
                    st.image(res_img, caption="Residual Error Distribution (Actual – Predicted)",
                             use_container_width=True)

            comp_img = os.path.join(PLOTS_DIR, "model_comparison.png")
            if os.path.exists(comp_img):
                st.image(comp_img, caption="Publication-Grade Metric Comparison Chart",
                         use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 5 — FEATURE DRIVERS & INTELLIGENCE
    # ══════════════════════════════════════════════════════════
    with tab_insights:
        st.subheader("🔍 Airfare Component Decomposition & Driver Rankings")
        st.write("Quantified percentage contribution of each feature using held-out permutation tests:")

        fi_col1, fi_col2 = st.columns([1.3, 1])

        with fi_col1:
            if importance_list:
                df_imp = pd.DataFrame(importance_list).sort_values(
                    "Importance_Percent", ascending=True
                )
                fig_imp = px.bar(
                    df_imp, x="Importance_Percent", y="Feature", orientation="h",
                    title="Relative Feature Importance Contribution (%)",
                    color="Importance_Percent",
                    color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#0ea5e9"], [1, "#38bdf8"]],
                    template="plotly_dark", text="Importance_Percent"
                )
                fig_imp.update_traces(
                    texttemplate="%{text:.1f}%", textposition="outside",
                    textfont=dict(color="#f8fafc")
                )
                fig_imp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=390, xaxis_title="Importance Share (%)", yaxis_title="Feature",
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#1e293b"),
                    margin=dict(l=10, r=40, t=40, b=10)
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        with fi_col2:
            if importance_list:
                st.markdown("##### 🏆 Ranked Component Impact")
                df_imp_display = pd.DataFrame(importance_list).sort_values(
                    "Importance_Percent", ascending=False
                )
                st.dataframe(
                    df_imp_display[['Feature', 'Importance_Percent']].rename(
                        columns={'Importance_Percent': 'Contribution (%)'}
                    ).style.format({'Contribution (%)': '{:.2f}%'})
                     .background_gradient(subset=['Contribution (%)'], cmap='Blues'),
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown("### 🛫 Pricing Insights & Yield Management Observations")
        st.markdown("""
        1. **Cabin Class Disparity (86.1% Weight):**
           Business class tickets trade at a structural multiple of **4×–6×** compared to
           economy, dominating variance in raw fare.

        2. **Booking Window (`days_left`) Non-Linear Dynamics:**
           Fares exhibit an exponential hockey-stick surge in the **final 7 days** before
           departure. Tickets purchased **20–35 days ahead** remain stable near floor pricing.

        3. **Airline Brand Segmentation:**
           Full-service carriers (Vistara, Air India) consistently show a premium pricing
           bracket, whereas budget carriers (IndiGo, AirAsia, SpiceJet, GO FIRST) compete
           aggressively on economy fare buckets.

        4. **Distance & Route Capacity:**
           Flight duration and route distance scale base operational costs linearly, while
           airport slots and ground handling tier impact the fixed taxes and fees non-linearly.
        """)

        days_img = os.path.join(PLOTS_DIR, "days_left_vs_price.png")
        if os.path.exists(days_img):
            st.image(days_img, caption="Booking Window vs Average Fare by Cabin Class",
                     use_container_width=True)
