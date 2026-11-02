"""
Comprehensive Indian Airports Database & Geodesic Navigation Engine.
Covers 50+ commercial airports across all Indian States and Union Territories.
Includes IATA codes, coordinates, airport tiers, and Haversine distance calculators.
"""

import math

AIRPORTS = {
    # Northern Region
    "Delhi": {
        "code": "DEL", "name": "Indira Gandhi International Airport",
        "state": "Delhi", "tier": "Metro", "lat": 28.5562, "lon": 77.1000, "nearest_hub": "Delhi"
    },
    "Jaipur": {
        "code": "JAI", "name": "Jaipur International Airport",
        "state": "Rajasthan", "tier": "Tier-2", "lat": 26.8242, "lon": 75.8122, "nearest_hub": "Delhi"
    },
    "Udaipur": {
        "code": "UDR", "name": "Maharana Pratap Airport",
        "state": "Rajasthan", "tier": "Tier-2", "lat": 24.6177, "lon": 73.8961, "nearest_hub": "Delhi"
    },
    "Jodhpur": {
        "code": "JDH", "name": "Jodhpur Airport",
        "state": "Rajasthan", "tier": "Tier-2", "lat": 26.2512, "lon": 73.0489, "nearest_hub": "Delhi"
    },
    "Lucknow": {
        "code": "LKO", "name": "Chaudhary Charan Singh International Airport",
        "state": "Uttar Pradesh", "tier": "Tier-2", "lat": 26.7606, "lon": 80.8893, "nearest_hub": "Delhi"
    },
    "Varanasi": {
        "code": "VNS", "name": "Lal Bahadur Shastri International Airport",
        "state": "Uttar Pradesh", "tier": "Tier-2", "lat": 25.4524, "lon": 82.8593, "nearest_hub": "Delhi"
    },
    "Chandigarh": {
        "code": "IXC", "name": "Shaheed Bhagat Singh International Airport",
        "state": "Punjab/Haryana", "tier": "Tier-2", "lat": 30.6735, "lon": 76.7885, "nearest_hub": "Delhi"
    },
    "Amritsar": {
        "code": "ATQ", "name": "Sri Guru Ram Dass Jee International Airport",
        "state": "Punjab", "tier": "Tier-2", "lat": 31.7096, "lon": 74.7973, "nearest_hub": "Delhi"
    },
    "Srinagar": {
        "code": "SXR", "name": "Sheikh ul-Alam International Airport",
        "state": "Jammu & Kashmir", "tier": "Tier-2", "lat": 33.9871, "lon": 74.7741, "nearest_hub": "Delhi"
    },
    "Jammu": {
        "code": "IXJ", "name": "Jammu Civil Enclave",
        "state": "Jammu & Kashmir", "tier": "Tier-2", "lat": 32.6891, "lon": 74.8374, "nearest_hub": "Delhi"
    },
    "Leh": {
        "code": "IXL", "name": "Kushok Bakula Rimpochee Airport",
        "state": "Ladakh", "tier": "Tier-2", "lat": 34.1359, "lon": 77.5465, "nearest_hub": "Delhi"
    },
    "Dehradun": {
        "code": "DED", "name": "Jolly Grant Airport",
        "state": "Uttarakhand", "tier": "Tier-2", "lat": 30.1897, "lon": 78.1803, "nearest_hub": "Delhi"
    },

    # Western Region
    "Mumbai": {
        "code": "BOM", "name": "Chhatrapati Shivaji Maharaj International Airport",
        "state": "Maharashtra", "tier": "Metro", "lat": 19.0896, "lon": 72.8656, "nearest_hub": "Mumbai"
    },
    "Pune": {
        "code": "PNQ", "name": "Pune Airport",
        "state": "Maharashtra", "tier": "Tier-2", "lat": 18.5822, "lon": 73.9197, "nearest_hub": "Mumbai"
    },
    "Nagpur": {
        "code": "NAG", "name": "Dr. Babasaheb Ambedkar International Airport",
        "state": "Maharashtra", "tier": "Tier-2", "lat": 21.0922, "lon": 79.0472, "nearest_hub": "Mumbai"
    },
    "Aurangabad": {
        "code": "IXU", "name": "Chhatrapati Sambhajinagar Airport",
        "state": "Maharashtra", "tier": "Tier-2", "lat": 19.8631, "lon": 75.3981, "nearest_hub": "Mumbai"
    },
    "Ahmedabad": {
        "code": "AMD", "name": "Sardar Vallabhbhai Patel International Airport",
        "state": "Gujarat", "tier": "Tier-2", "lat": 23.0734, "lon": 72.6347, "nearest_hub": "Mumbai"
    },
    "Surat": {
        "code": "STV", "name": "Surat International Airport",
        "state": "Gujarat", "tier": "Tier-2", "lat": 21.1141, "lon": 72.7419, "nearest_hub": "Mumbai"
    },
    "Vadodara": {
        "code": "BDQ", "name": "Vadodara Airport",
        "state": "Gujarat", "tier": "Tier-2", "lat": 22.3362, "lon": 73.2263, "nearest_hub": "Mumbai"
    },
    "Goa (Dabolim/Mopa)": {
        "code": "GOI", "name": "Dabolim / Manohar International Airport",
        "state": "Goa", "tier": "Tier-2", "lat": 15.3808, "lon": 73.8314, "nearest_hub": "Mumbai"
    },
    "Indore": {
        "code": "IDR", "name": "Devi Ahilya Bai Holkar Airport",
        "state": "Madhya Pradesh", "tier": "Tier-2", "lat": 22.7217, "lon": 75.8011, "nearest_hub": "Mumbai"
    },
    "Bhopal": {
        "code": "BHO", "name": "Raja Bhoj Airport",
        "state": "Madhya Pradesh", "tier": "Tier-2", "lat": 23.2875, "lon": 77.3374, "nearest_hub": "Mumbai"
    },

    # Southern Region
    "Bangalore": {
        "code": "BLR", "name": "Kempegowda International Airport",
        "state": "Karnataka", "tier": "Metro", "lat": 13.1986, "lon": 77.7066, "nearest_hub": "Bangalore"
    },
    "Mangalore": {
        "code": "IXE", "name": "Mangaluru International Airport",
        "state": "Karnataka", "tier": "Tier-2", "lat": 12.9613, "lon": 74.8901, "nearest_hub": "Bangalore"
    },
    "Hubli": {
        "code": "HBX", "name": "Hubballi Airport",
        "state": "Karnataka", "tier": "Tier-3", "lat": 15.3617, "lon": 75.0849, "nearest_hub": "Bangalore"
    },
    "Belgaum": {
        "code": "IXG", "name": "Belagavi Airport",
        "state": "Karnataka", "tier": "Tier-3", "lat": 15.8593, "lon": 74.6183, "nearest_hub": "Bangalore"
    },
    "Hyderabad": {
        "code": "HYD", "name": "Rajiv Gandhi International Airport",
        "state": "Telangana", "tier": "Metro", "lat": 17.2403, "lon": 78.4294, "nearest_hub": "Hyderabad"
    },
    "Visakhapatnam": {
        "code": "VTZ", "name": "Visakhapatnam Airport",
        "state": "Andhra Pradesh", "tier": "Tier-2", "lat": 17.7212, "lon": 83.2245, "nearest_hub": "Hyderabad"
    },
    "Vijayawada": {
        "code": "VGA", "name": "Vijayawada International Airport",
        "state": "Andhra Pradesh", "tier": "Tier-2", "lat": 16.5304, "lon": 80.7968, "nearest_hub": "Hyderabad"
    },
    "Tirupati": {
        "code": "TIR", "name": "Tirupati Airport",
        "state": "Andhra Pradesh", "tier": "Tier-2", "lat": 13.6325, "lon": 79.5434, "nearest_hub": "Chennai"
    },
    "Chennai": {
        "code": "MAA", "name": "Chennai International Airport",
        "state": "Tamil Nadu", "tier": "Metro", "lat": 12.9941, "lon": 80.1709, "nearest_hub": "Chennai"
    },
    "Coimbatore": {
        "code": "CJB", "name": "Coimbatore International Airport",
        "state": "Tamil Nadu", "tier": "Tier-2", "lat": 11.0299, "lon": 77.0434, "nearest_hub": "Chennai"
    },
    "Madurai": {
        "code": "IXM", "name": "Madurai Airport",
        "state": "Tamil Nadu", "tier": "Tier-2", "lat": 9.8345, "lon": 78.0934, "nearest_hub": "Chennai"
    },
    "Tiruchirappalli": {
        "code": "TRZ", "name": "Tiruchirappalli International Airport",
        "state": "Tamil Nadu", "tier": "Tier-2", "lat": 10.7654, "lon": 78.7097, "nearest_hub": "Chennai"
    },
    "Kochi": {
        "code": "COK", "name": "Cochin International Airport",
        "state": "Kerala", "tier": "Tier-2", "lat": 10.1556, "lon": 76.4019, "nearest_hub": "Bangalore"
    },
    "Thiruvananthapuram": {
        "code": "TRV", "name": "Thiruvananthapuram International Airport",
        "state": "Kerala", "tier": "Tier-2", "lat": 8.4821, "lon": 76.9200, "nearest_hub": "Chennai"
    },
    "Calicut": {
        "code": "CCJ", "name": "Calicut International Airport",
        "state": "Kerala", "tier": "Tier-2", "lat": 11.1368, "lon": 75.9553, "nearest_hub": "Bangalore"
    },

    # Eastern & North-Eastern Region
    "Kolkata": {
        "code": "CCU", "name": "Netaji Subhas Chandra Bose International Airport",
        "state": "West Bengal", "tier": "Metro", "lat": 22.6547, "lon": 88.4467, "nearest_hub": "Kolkata"
    },
    "Bagdogra (Siliguri)": {
        "code": "IXB", "name": "Bagdogra Airport",
        "state": "West Bengal", "tier": "Tier-2", "lat": 26.6812, "lon": 88.3286, "nearest_hub": "Kolkata"
    },
    "Bhubaneswar": {
        "code": "BBI", "name": "Biju Patnaik International Airport",
        "state": "Odisha", "tier": "Tier-2", "lat": 20.2444, "lon": 85.8178, "nearest_hub": "Kolkata"
    },
    "Patna": {
        "code": "PAT", "name": "Jay Prakash Narayan International Airport",
        "state": "Bihar", "tier": "Tier-2", "lat": 25.5913, "lon": 85.0880, "nearest_hub": "Kolkata"
    },
    "Ranchi": {
        "code": "IXR", "name": "Birsa Munda Airport",
        "state": "Jharkhand", "tier": "Tier-2", "lat": 23.3143, "lon": 85.3216, "nearest_hub": "Kolkata"
    },
    "Raipur": {
        "code": "RPR", "name": "Swami Vivekananda Airport",
        "state": "Chhattisgarh", "tier": "Tier-2", "lat": 21.1804, "lon": 81.7388, "nearest_hub": "Hyderabad"
    },
    "Guwahati": {
        "code": "GAU", "name": "Lokpriya Gopinath Bordoloi International Airport",
        "state": "Assam", "tier": "Tier-2", "lat": 26.1061, "lon": 91.5859, "nearest_hub": "Kolkata"
    },
    "Dibrugarh": {
        "code": "DIB", "name": "Dibrugarh Airport",
        "state": "Assam", "tier": "Tier-3", "lat": 27.4839, "lon": 95.0197, "nearest_hub": "Kolkata"
    },
    "Agartala": {
        "code": "IXA", "name": "Maharaja Bir Bikram Airport",
        "state": "Tripura", "tier": "Tier-2", "lat": 23.8869, "lon": 91.2405, "nearest_hub": "Kolkata"
    },
    "Imphal": {
        "code": "IMF", "name": "Bir Tikendrajit International Airport",
        "state": "Manipur", "tier": "Tier-2", "lat": 24.7600, "lon": 93.8967, "nearest_hub": "Kolkata"
    },
    "Dimapur": {
        "code": "DMU", "name": "Dimapur Airport",
        "state": "Nagaland", "tier": "Tier-3", "lat": 25.8837, "lon": 93.7712, "nearest_hub": "Kolkata"
    },
    "Port Blair": {
        "code": "IXZ", "name": "Veer Savarkar International Airport",
        "state": "Andaman & Nicobar", "tier": "Tier-2", "lat": 11.6410, "lon": 92.7297, "nearest_hub": "Chennai"
    }
}

METRO_HUBS = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

def get_route_details(src_city: str, dst_city: str):
    """
    Computes distance, bearing, estimated duration, and routing metadata between any two Indian cities.
    """
    src = AIRPORTS.get(src_city)
    dst = AIRPORTS.get(dst_city)

    if not src or not dst:
        return None

    distance_km = haversine_distance_km(src['lat'], src['lon'], dst['lat'], dst['lon'])

    # Determine flight characteristics based on distance
    # Typical passenger jet: ~650 km/h block speed + 25-30 min taxi/climb/approach
    est_duration_hours = max(0.9, round((distance_km / 650.0) + 0.45, 1))

    # Recommended stops
    is_both_metros = (src_city in METRO_HUBS and dst_city in METRO_HUBS)
    if is_both_metros:
        rec_stops = "zero"
        proxy_src = src_city
        proxy_dst = dst_city
    else:
        # If long distance or small tiers, 1 stop via hub is common
        if distance_km > 1200 or src['tier'] == 'Tier-3' or dst['tier'] == 'Tier-3':
            rec_stops = "one"
            est_duration_hours = round(est_duration_hours + 1.8, 1) # add layover time
        else:
            rec_stops = "zero"

        proxy_src = src.get('nearest_hub', 'Delhi')
        proxy_dst = dst.get('nearest_hub', 'Mumbai')
        if proxy_src == proxy_dst:
            # If same regional hub, route to another major hub
            proxy_dst = 'Delhi' if proxy_src != 'Delhi' else 'Mumbai'

    # Bearing angle (degrees)
    lat1, lon1 = math.radians(src['lat']), math.radians(src['lon'])
    lat2, lon2 = math.radians(dst['lat']), math.radians(dst['lon'])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    # Carbon emissions estimate (~0.115 kg CO2 per passenger-km)
    co2_kg = round(distance_km * 0.115, 1)

    return {
        "src": src,
        "dst": dst,
        "distance_km": distance_km,
        "distance_miles": round(distance_km * 0.621371, 1),
        "est_duration": est_duration_hours,
        "rec_stops": rec_stops,
        "proxy_src": proxy_src,
        "proxy_dst": proxy_dst,
        "bearing": round(bearing, 1),
        "co2_kg": co2_kg,
        "is_direct_dataset_route": is_both_metros
    }

def get_pricing_breakdown(total_fare: float, flight_class: str, src_tier: str, dst_tier: str):
    """
    Decomposes the final airfare into real aviation industry fare components:
    - Base Fare
    - Fuel Surcharge (YQ/YR)
    - User Development Fee (UDF)
    - Passenger Service Fee (PSF) & Aviation Security Fee (ASF)
    - Goods & Services Tax (GST: 5% Economy, 12% Business)
    """
    gst_rate = 0.05 if flight_class == "Economy" else 0.12
    # Pre-tax base
    subtotal = total_fare / (1 + gst_rate)
    gst_amount = total_fare - subtotal

    # Airport fees based on airport tiers
    udf_origin = 450 if src_tier == "Metro" else 250
    udf_dest = 450 if dst_tier == "Metro" else 250
    asf_fee = 236 # Standard DGCA aviation security fee in India

    airport_fees = udf_origin + udf_dest + asf_fee
    remaining = max(1000.0, subtotal - airport_fees)

    fuel_surcharge = round(remaining * 0.28, 2)
    base_fare = round(remaining - fuel_surcharge, 2)

    return {
        "Base Fare": base_fare,
        "Fuel & Dynamic Surcharge (YQ)": fuel_surcharge,
        "Airport Development Fee (UDF/PSF)": airport_fees,
        "Goods & Services Tax (GST)": round(gst_amount, 2),
        "Total Estimated Fare": round(total_fare, 2)
    }
