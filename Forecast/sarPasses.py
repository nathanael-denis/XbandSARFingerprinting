import requests
import json
from datetime import datetime, timezone, timedelta

'''
This script will provide all radio passes in the next 3 days over your location, for all ICEYE Satellites including those who do not operate anymore 
You need two update three lines to adjust to your location:

API_KEY : I can not provide my own API key so your need to get your own on N2YO (free and straightforward)
Latitude and longitude : observer_lat and observer_lng

After you collect the radio passes, you can put them in a more human-readable format by running refinePasses.py 

'''
def get_radio_passes(norad_id, observer_lat, observer_lng, observer_alt, days, min_elevation, api_key):
    base_url = "https://api.n2yo.com/rest/v1/satellite/radiopasses"
    url = f"{base_url}/{norad_id}/{observer_lat}/{observer_lng}/{observer_alt}/{days}/{min_elevation}/&apiKey={api_key}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Convert Unix timestamps to human-readable dates
        for passes in data.get("passes", []):
            passes["startUTC_readable"] = unix_to_readable(passes["startUTC"])
            passes["maxUTC_readable"] = unix_to_readable(passes["maxUTC"])
            passes["endUTC_readable"] = unix_to_readable(passes["endUTC"])
        return data
    else:
        print(f"Error: Unable to fetch data for NORAD ID {norad_id}. HTTP Status Code: {response.status_code}")
        return None

def unix_to_readable(timestamp):
    # Convert to UTC time
    dt_utc = datetime.utcfromtimestamp(timestamp).replace(tzinfo=timezone.utc)
    # Convert to local time (UTC+3)
    dt_local = dt_utc.astimezone(timezone(timedelta(hours=3)))
    return dt_local.strftime('%Y-%m-%d %H:%M:%S')

def write_passes_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


#ICEYE Ids

iceye_norad_ids = [
    43114,  # ICEYE-X1
    43800,  # ICEYE-X2
    43800,  # ICEYE-X3 (Part of Harbinger)
    44390,  # ICEYE-X4
    44389,  # ICEYE-X5
    46497,  # ICEYE-X6
    46496,  # ICEYE-X7
    47510,  # ICEYE-X8
    47506,  # ICEYE-X9
    47507,  # ICEYE-X10 (XR-1)
    48918,  # ICEYE-X11
    48914,  # ICEYE-X12
    48916,  # ICEYE-X13
    51070,  # ICEYE-X14
    48917,  # ICEYE-X15
    51008,  # ICEYE-X16
    52762,  # ICEYE-X17
    52749,  # ICEYE-X18
    52758,  # ICEYE-X19
    52759,  # ICEYE-X20
    55049,  # ICEYE-X21
    52755,  # ICEYE-X24
    56963,  # ICEYE-X25
    56961,  # ICEYE-X26
    55062,  # ICEYE-X27
    56947,  # ICEYE-X30
    56949,  # ICEYE-X23
    58288,  # ICEYE-X31
    58293,  # ICEYE-X32
    60548,  # ICEYE-X33
    58294,  # ICEYE-X34
    58302,  # ICEYE-X35
    59103,  # ICEYE-X36
    59102,  # ICEYE-X37
    59100,  # ICEYE-X38
    60546,  # ICEYE-X39
    60549,  # ICEYE-X40
    60539,  # ICEYE-X43
    62389,  # ICEYE-X47
    62384,   # ICEYE-X49
    62698,   # ICEYE-X42
    62700,   # ICEYE-X41
    62707,   # ICEYE-X44
    62705,  # ICEYE-X45
    63258, # ICEYE-X46
    63253, # ICEYE-X48
    63255, # ICEYE-X50
    63257, # ICEYE-X51
]


# Parameters

######## !!!! #########
observer_lat = 11.1111111      #ADJUST TO YOUR LOCATION
observer_lng = 11.1111111      #ADJUST TO YOUR LOCATION

observer_alt = 10  
days = 3  # Number of days of prediction (max 10)
min_elevation = 10  # Minimum elevation acceptable for the highest altitude point of the pass (degrees)
api_key = "AAAAAA-BBBBBB-CCCCCC-DDDDDD"  # N2YO API key PUT YOUR OWN KEY HERE

all_passes = {}

# Get radio passes for each Gaofen satellite
for norad_id in iceye_norad_ids:
    passes = get_radio_passes(norad_id, observer_lat, observer_lng, observer_alt, days, min_elevation, api_key)
    if passes:
        all_passes[norad_id] = passes

# Write all passes to file
if all_passes:
    write_passes_to_file(all_passes, "radio_passes.txt")
    print("Radio passes for all Gaofen satellites written to radio_passes.txt")