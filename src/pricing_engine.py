"""
Advanced Aviation Pricing & Yield Management Engine for Nationwide Domestic Flights.
Integrates ML models with Great-Circle Geodesics, Multi-Carrier Benchmarking,
Component Cost Breakdown, and Smart Booking Window Advisory.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from airports_data import AIRPORTS, METRO_HUBS, get_route_details, get_pricing_breakdown

CARRIERS = [
    {"name": "IndiGo", "code": "6E", "type": "Low-Cost Carrier", "model_key": "Indigo", "badge": "Best On-Time"},
    {"name": "Air India", "code": "AI", "type": "Full-Service Legacy", "model_key": "Air_India", "badge": "Free Meals & Baggage"},
    {"name": "Vistara", "code": "UK", "type": "Full-Service Premium", "model_key": "Vistara", "badge": "Top Rated Cabin"},
    {"name": "AirAsia India", "code": "I5", "type": "Low-Cost Carrier", "model_key": "AirAsia", "badge": "Budget Friendly"},
    {"name": "SpiceJet", "code": "SG", "type": "Low-Cost Carrier", "model_key": "SpiceJet", "badge": "SpiceMax Options"},
    {"name": "GO FIRST", "code": "G8", "type": "Low-Cost Carrier", "model_key": "GO_FIRST", "badge": "Ultra Budget"}
]

def load_inference_pipeline(base_dir: str):
    model_path = os.path.join(base_dir, "models", "best_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def predict_single_flight(pipeline, src_city: str, dst_city: str, airline_name: str,
                          flight_class: str, dep_time: str, arr_time: str, days_left: int):
    """
    Predicts single flight fare with distance scaling and component breakdown.
    """
    route_info = get_route_details(src_city, dst_city)
    if not route_info or pipeline is None:
        return None

    proxy_src = route_info['proxy_src']
    proxy_dst = route_info['proxy_dst']
    est_duration = route_info['est_duration']
    stops = route_info['rec_stops']

    # Normalize carrier name
    matched_carrier = next((c for c in CARRIERS if c['name'] == airline_name), None)
    model_carrier = matched_carrier['model_key'] if matched_carrier else 'Indigo'

    input_df = pd.DataFrame([{
        'airline': model_carrier,
        'source_city': proxy_src,
        'destination_city': proxy_dst,
        'departure_time': dep_time,
        'arrival_time': arr_time,
        'stops': stops,
        'class': flight_class,
        'route': f"{proxy_src} -> {proxy_dst}",
        'duration': est_duration,
        'days_left': days_left
    }])

    raw_pred = float(pipeline.predict(input_df)[0])

    if route_info['is_direct_dataset_route']:
        final_price = raw_pred
    else:
        # Distance scaling & regional tier adjustment
        proxy_info = get_route_details(proxy_src, proxy_dst)
        proxy_dist = proxy_info['distance_km'] if proxy_info else 1150.0
        dist_ratio = route_info['distance_km'] / max(300.0, proxy_dist)
        taper_factor = dist_ratio ** 0.42

        tier_mult = 1.0
        if route_info['src']['tier'] in ['Tier-2', 'Tier-3']:
            tier_mult += 0.05
        if route_info['dst']['tier'] in ['Tier-2', 'Tier-3']:
            tier_mult += 0.05

        final_price = raw_pred * taper_factor * tier_mult

    final_price = max(1800.0, round(final_price, -1))
    breakdown = get_pricing_breakdown(final_price, flight_class, route_info['src']['tier'], route_info['dst']['tier'])

    return {
        'price': final_price,
        'route_info': route_info,
        'breakdown': breakdown
    }

def get_multi_carrier_comparison(pipeline, src_city: str, dst_city: str,
                                 flight_class: str, dep_time: str, arr_time: str, days_left: int):
    """
    Evaluates pricing across all 6 airlines simultaneously to create a comparison matrix.
    """
    comparison = []
    for carrier in CARRIERS:
        res = predict_single_flight(
            pipeline, src_city, dst_city, carrier['name'], flight_class, dep_time, arr_time, days_left
        )
        if res:
            comparison.append({
                'Airline': carrier['name'],
                'Code': carrier['code'],
                'Type': carrier['type'],
                'Estimated_Fare': res['price'],
                'Duration_Hours': res['route_info']['est_duration'],
                'Stops': res['route_info']['rec_stops'].capitalize().replace('_', ' '),
                'Badge': carrier['badge']
            })

    df_comp = pd.DataFrame(comparison).sort_values(by='Estimated_Fare').reset_index(drop=True)
    return df_comp

def get_booking_advisory(days_left: int, current_price: float, flight_class: str):
    """
    Yield management heuristic advisor informing users whether to BUY NOW or WAIT.
    """
    if days_left <= 3:
        status = "CRITICAL SURGE (BOOK IMMEDIATELY)"
        color = "#e63946"
        icon = "🚨"
        action = "Prices are currently in peak surge due to last-minute distress yield pricing. Fares will only rise or sell out."
        expected_trend = "Rising +15% to +35% every 24 hours."
    elif 4 <= days_left <= 7:
        status = "HIGH DEMAND SURGE"
        color = "#f4a261"
        icon = "⚠️"
        action = "You are entering the high-urgency booking band. Booking today will avoid steep 72-hour price jumps."
        expected_trend = "Likely to climb steeply as departure nears."
    elif 8 <= days_left <= 21:
        status = "FAIR VALUE WINDOW"
        color = "#2a9d8f"
        icon = "✅"
        action = "Pricing is currently stable. Good balance between schedule certainty and reasonable tariff."
        expected_trend = "Moderate stability; slight upward drift."
    elif 22 <= days_left <= 38:
        status = "OPTIMAL SWEET SPOT (BEST TIME TO BUY)"
        color = "#2b9348"
        icon = "🎯"
        action = "Historically the best price-to-seat-availability window! Airlines have released promotional booking classes."
        expected_trend = "At lowest floor baseline."
    else:
        status = "EARLY BIRD (MONITOR FOR 1-2 WEEKS)"
        color = "#457b9d"
        icon = "⏳"
        action = "You have plenty of time. Airlines often release targeted seat flash-sales 30-40 days prior to travel."
        expected_trend = "Prices remain stable or may slightly soften during mid-month sales."

    return {
        "status": status,
        "color": color,
        "icon": icon,
        "action": action,
        "expected_trend": expected_trend
    }
