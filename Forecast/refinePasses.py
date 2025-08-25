import pytz
import json
from datetime import datetime, timezone, timedelta

def process_passes_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        data = json.load(file)

    passes_list = []

    # Extract necessary information and append to passes_list
    for sat_data in data.values():
        satname = sat_data["info"]["satname"]

        # Check if 'passes' key exists
        if "passes" in sat_data:
            for pass_info in sat_data["passes"]:
                startUTC_readable = pass_info["startUTC_readable"]
                endUTC_readable = pass_info["endUTC_readable"]
                passes_list.append((startUTC_readable, endUTC_readable, satname))

    # Sort the list by startUTC_readable
    passes_list.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'))

    # Write the sorted passes to the output file with day separators
    with open(output_filename, 'w') as file:
        current_day = ""
        for pass_info in passes_list:
            start_date = pass_info[0].split(' ')[0]
            if start_date != current_day:
                if current_day != "":
                    file.write("-------------------\n")
                current_day = start_date
            file.write(f"{pass_info[0]}, {pass_info[1]}, {pass_info[2]}\n")

# Process the radio_passes.txt to derive another text file in the desired structure
process_passes_file("radio_passes.txt", "sorted_radio_passes.txt")
print("Sorted radio passes written to sorted_radio_passes.txt")
