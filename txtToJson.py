import json
import os

#gets script directory
script_directory = os.path.dirname(os.path.abspath(__file__))

#sets the working directory 
os.chdir(script_directory)

file_path = "cleanedData.txt"

#reads text from cleanedData.txt
with open("cleanedData.txt", "r", encoding="utf-8")as file:
    lines = file.readlines()

#formatting data
data = [{"input": line.strip(), "output": line.strip()} for line in lines]

#ssave file in same directoru
json_file_path = os.path.join(script_directory, "formattedData.json")

with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data successfully converted to JSON format and saved at: {json_file_path}")
