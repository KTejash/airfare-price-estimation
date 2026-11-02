"""
Real-Time Flight Price Fetcher via Google Flights Engine.
Retrieves real-time live market tariffs, carrier options, and schedules.
Provides comparison metrics between ML predicted baseline and live OTA market prices.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# In-memory query cache with TTL (15 minutes)
_REALTIME_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 900


def get_cache_key(src_code: str, dst_code: str, target_date: str, cabin_class: str) -> str:
    return f"{src_code}_{dst_code}_{target_date}_{cabin_class.lower()}"


def fetch_realtime_flights(
    src_code: str,
    dst_code: str,
    days_left: int = 14,
    cabin_class: str = "Economy",
    timeout_sec: int = 10
) -> Dict[str, Any]:
    """
    Queries live real-time flight tariffs from Google Flights for the given route and departure date.
    Returns:
        dict with status, timestamp, flight list, lowest carrier fares, and metadata.
    """
    import fast_flights

    # Calculate travel date from days_left
    dep_date = datetime.now() + timedelta(days=max(1, days_left))
    date_str = dep_date.strftime("%Y-%m-%d")
    cache_key = get_cache_key(src_code, dst_code, date_str, cabin_class)

    # Check cache first
    now = time.time()
    if cache_key in _REALTIME_CACHE:
        entry = _REALTIME_CACHE[cache_key]
        if now - entry["timestamp"] < CACHE_TTL_SECONDS:
            cached_data = dict(entry["data"])
            cached_data["cached"] = True
            return cached_data

    seat_type = "business" if cabin_class.lower() == "business" else "economy"

    try:
        query = fast_flights.create_query(
            flights=[fast_flights.FlightQuery(
                date=date_str,
                from_airport=src_code.upper(),
                to_airport=dst_code.upper()
            )],
            seat=seat_type,
            trip="one-way",
            currency="INR"
        )

        res = fast_flights.get_flights(query)
        flight_items: List[Dict[str, Any]] = []
        carrier_best: Dict[str, Dict[str, Any]] = {}

        if res and len(res) > 0:
            for f in res:
                price = getattr(f, "price", 0)
                if not price or price <= 0:
                    continue

                airlines = getattr(f, "airlines", [])
                primary_airline = airlines[0] if airlines else "Other Carrier"
                flight_legs = getattr(f, "flights", [])
                stops_count = max(0, len(flight_legs) - 1)
                total_duration_min = sum(getattr(leg, "duration", 0) for leg in flight_legs) if flight_legs else 0

                # Extract departure and arrival formatted strings if available
                dep_time_str = ""
                arr_time_str = ""
                aircraft = ""
                if flight_legs:
                    first_leg = flight_legs[0]
                    last_leg = flight_legs[-1]
                    dep_obj = getattr(first_leg, "departure", None)
                    arr_obj = getattr(last_leg, "arrival", None)
                    aircraft = getattr(first_leg, "plane_type", "")

                    if dep_obj and hasattr(dep_obj, "time"):
                        dep_time_str = f"{dep_obj.time[0]:02d}:{dep_obj.time[1]:02d}"
                    if arr_obj and hasattr(arr_obj, "time"):
                        arr_time_str = f"{arr_obj.time[0]:02d}:{arr_obj.time[1]:02d}"

                item = {
                    "airline": primary_airline,
                    "price": price,
                    "stops": stops_count,
                    "stops_label": "Non-stop" if stops_count == 0 else f"{stops_count} stop",
                    "duration_hours": round(total_duration_min / 60.0, 1) if total_duration_min else 2.0,
                    "departure_time": dep_time_str or "Flexible",
                    "arrival_time": arr_time_str or "Flexible",
                    "aircraft": aircraft or "Boeing / Airbus",
                    "source": "Google Flights (Live)"
                }
                flight_items.append(item)

                # Keep track of lowest fare per carrier
                if primary_airline not in carrier_best or price < carrier_best[primary_airline]["price"]:
                    carrier_best[primary_airline] = item

            # Sort all flights by lowest price
            flight_items.sort(key=lambda x: x["price"])
            lowest_overall = flight_items[0]["price"] if flight_items else None

            result_data = {
                "success": True,
                "cached": False,
                "source": "Google Flights Live Feed",
                "date_str": date_str,
                "flight_count": len(flight_items),
                "lowest_price": lowest_overall,
                "carrier_best": carrier_best,
                "flights": flight_items[:25],  # top 25 live flights
                "error": None
            }

            # Cache the result
            _REALTIME_CACHE[cache_key] = {
                "timestamp": now,
                "data": result_data
            }
            return result_data

        else:
            return {
                "success": False,
                "cached": False,
                "source": "Google Flights Live Feed",
                "date_str": date_str,
                "flight_count": 0,
                "lowest_price": None,
                "carrier_best": {},
                "flights": [],
                "error": "No real-time flights found for this specific date and route on Google Flights."
            }

    except Exception as e:
        return {
            "success": False,
            "cached": False,
            "source": "Google Flights Live Feed",
            "date_str": date_str,
            "flight_count": 0,
            "lowest_price": None,
            "carrier_best": {},
            "flights": [],
            "error": f"Live fetch exception: {str(e)}"
        }


def compute_market_variance(ml_predicted: float, live_price: Optional[float]) -> Dict[str, Any]:
    """
    Computes delta and variance between the Supervised ML prediction and live Google Flights price.
    """
    if not live_price or live_price <= 0:
        return {
            "has_live": False,
            "difference": 0.0,
            "variance_pct": 0.0,
            "status": "ML Baseline (No Live Comparison)",
            "badge_color": "#64748b"
        }

    diff = live_price - ml_predicted
    pct = (diff / ml_predicted) * 100.0

    if abs(pct) <= 6.0:
        status = "🎯 Near-Exact ML Match (±6%)"
        badge_col = "#10b981"
    elif diff > 0:
        status = f"📈 Live Market Premium (+{pct:.1f}%)"
        badge_col = "#fbbf24"
    else:
        status = f"📉 Live OTA Discount ({pct:.1f}%)"
        badge_col = "#38bdf8"

    return {
        "has_live": True,
        "difference": round(diff, 2),
        "variance_pct": round(pct, 1),
        "status": status,
        "badge_color": badge_col,
        "live_price": live_price,
        "ml_price": ml_predicted
    }
