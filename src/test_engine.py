import os
import sys
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from airports_data import AIRPORTS, get_route_details, get_pricing_breakdown

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.joblib")
model = joblib.load(model_path)

def estimate_nationwide_fare(src_city, dst_city, airline, flight_class, dep_time, arr_time, days_left):
    route_info = get_route_details(src_city, dst_city)
    if not route_info:
        return None

    proxy_src = route_info['proxy_src']
    proxy_dst = route_info['proxy_dst']
    est_duration = route_info['est_duration']
    stops = route_info['rec_stops']

    # Map airline name to dataset representation
    airline_map = {
        'IndiGo': 'Indigo',
        'Air India': 'Air_India',
        'GO FIRST': 'GO_FIRST',
        'Vistara': 'Vistara',
        'SpiceJet': 'SpiceJet',
        'AirAsia': 'AirAsia'
    }
    clean_airline = airline_map.get(airline, airline)

    input_df = pd.DataFrame([{
        'airline': clean_airline,
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

    raw_pred = float(model.predict(input_df)[0])

    if route_info['is_direct_dataset_route']:
        final_price = raw_pred
    else:
        # Distance scaling
        proxy_info = get_route_details(proxy_src, proxy_dst)
        proxy_dist = proxy_info['distance_km'] if proxy_info else 1150.0
        dist_ratio = route_info['distance_km'] / max(300.0, proxy_dist)
        taper_factor = dist_ratio ** 0.42

        # Tier premium for non-metro operations
        tier_mult = 1.0
        if route_info['src']['tier'] in ['Tier-2', 'Tier-3']:
            tier_mult += 0.05
        if route_info['dst']['tier'] in ['Tier-2', 'Tier-3']:
            tier_mult += 0.05

        final_price = raw_pred * taper_factor * tier_mult

    final_price = max(1800.0, round(final_price, -1)) # clean rounded rupees
    breakdown = get_pricing_breakdown(final_price, flight_class, route_info['src']['tier'], route_info['dst']['tier'])

    return {
        'route_info': route_info,
        'final_price': final_price,
        'breakdown': breakdown
    }

if __name__ == "__main__":
    test_cases = [
        ("Delhi", "Mumbai", "Vistara", "Economy", 20),
        ("Goa (Dabolim/Mopa)", "Srinagar", "IndiGo", "Economy", 14),
        ("Jaipur", "Bangalore", "Air India", "Business", 7),
        ("Kochi", "Guwahati", "AirAsia", "Economy", 3),
    ]

    for src, dst, al, cl, dl in test_cases:
        res = estimate_nationwide_fare(src, dst, al, cl, "Morning", "Evening", dl)
        print(f"{src} -> {dst} | {al} | {cl} | {dl} days left:")
        print(f"  Distance: {res['route_info']['distance_km']} km | Duration: {res['route_info']['est_duration']}h | Stops: {res['route_info']['rec_stops']}")
        print(f"  Estimated Fare: INR {res['final_price']:,.0f}")
        print(f"  Breakdown: Base={res['breakdown']['Base Fare']} | Fuel={res['breakdown']['Fuel & Dynamic Surcharge (YQ)']} | Tax={res['breakdown']['Goods & Services Tax (GST)']}")
        print("-" * 65)
