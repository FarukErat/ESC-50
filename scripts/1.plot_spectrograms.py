import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_and_save_spectrogram(audio_path, output_image_path):
    y, sr = librosa.load(audio_path, sr=None)

    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_axis_off()
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap="magma")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_all_wav_files_in_folder(folder_path):
    spectrogram_folder = os.path.join("spectrograms")
    if not os.path.exists(spectrogram_folder):
        os.makedirs(spectrogram_folder)

    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

    for filename in tqdm(wav_files, desc="Processing files"):
        audio_file_path = os.path.join(folder_path, filename)
        output_image_name = os.path.splitext(filename)[0] + "_spectrogram.png"
        output_image_path = os.path.join(spectrogram_folder, output_image_name)
        plot_and_save_spectrogram(audio_file_path, output_image_path)

if __name__ == "__main__":
    process_all_wav_files_in_folder("audio")
