import json
import os
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Parse JSON file
            return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return None
    

json_file_path = 'example_json.json'

# Read the JSON data
json_data = read_json_file(json_file_path)

if json_data:
    # Print the states
    print("States:")
    for state in json_data['states']:
        pos = state['pos']
        time = state['time']
        print(f"pos: {pos}, time: {time}")