import os
import csv

csv_file = 'meta/esc50.csv'
audio_folder = 'audio'

with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        original_filename = row['filename']
        category = row['category']
        new_filename = f"{category}_{original_filename}"

        src = os.path.join(audio_folder, original_filename)
        dst = os.path.join(audio_folder, new_filename)

        if os.path.exists(src):
            os.rename(src, dst)
            print(f"Renamed: {original_filename} → {new_filename}")
        else:
            print(f"File not found: {original_filename}")
