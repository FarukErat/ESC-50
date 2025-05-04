import pandas as pd
import os

df = pd.read_csv('meta/esc50.csv')

audio_folder = 'audio'

for _, row in df.iterrows():
    original_name = row['filename']
    new_name = f"{row['category']}_{row['src_file']}.wav"

    src_path = os.path.join(audio_folder, original_name)
    dst_path = os.path.join(audio_folder, new_name)

    if os.path.exists(src_path):
        os.rename(src_path, dst_path)
        print(f"Renamed: {original_name} → {new_name}")
    else:
        print(f"File not found: {original_name}")
